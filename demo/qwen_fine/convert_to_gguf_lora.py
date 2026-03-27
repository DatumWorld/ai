# 加载原始模型 + 挂载你的 LoRA
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3.5-9B-bnb-4bit", # 原始模型
    max_seq_length = 2048,
    load_in_4bit = True,
)
model = FastLanguageModel.get_peft_model(model, r = 16) # 结构需一致
model.load_adapter("./your_lora_path") # 加载你的微调权重

# 然后再执行 save_pretrained_gguf...
