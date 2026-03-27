# 训练程序
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_DATASETS_OFFLINE"] = "1" # <--- 关键参数：强制只读本地
os.environ["TRANSFORMERS_OFFLINE"] = "1" # <--- 关键参数：强制只读本地


from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/path/to/unsloth_Qwen3.5-9B", # 或使用 Instruct 版本unsloth/Qwen3.5-9B
    max_seq_length = 2048,           # 推荐 2048，最高支持 40960
    load_in_4bit = True,             # 显著减少显存占用
local_files_only = True, # <--- 关键参数：强制只读本地
)



model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # 建议值：8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # 优化为 0 以获得更好性能
    bias = "none",    # 优化为 "none"
    use_gradient_checkpointing = "unsloth", # 节省显存的关键
)




from datasets import load_dataset

# 1. 加载本地数据
dataset = load_dataset("json", data_files={"train": "mini_dataset.jsonl"}, split="train")

# 2. 定义 Qwen 的 Prompt 模板
# Qwen3.5 通常使用 <|im_start|> 格式，但为了简单微调，我们可以直接拼成字符串
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # 记得在结尾加上 EOS token，否则模型会停不下来
        text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

# 3. 映射数据集
dataset = dataset.map(formatting_prompts_func, batched = True)



from trl import SFTTrainer
from transformers import TrainingArguments

# 为2080ti 22GB 优化
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 1024,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        max_steps = 60, # 根据需求调整
        learning_rate = 2e-4,
        fp16 = True,
        bf16 = False,
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        weight_decay = 0.01,
    ),
)
trainer.train()


model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
