# 模型格式转换
from unsloth import FastLanguageModel
import torch

# 1. 指向你训练完保存的 safetensors 文件夹路径
# 注意：该文件夹下必须包含 config.json, tokenizer.json 等文件
local_model_path = "./model"

# 2. 重新加载模型（开启 4bit 以节省转换时的内存占用）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = local_model_path,
    max_seq_length=128,  # 笔记本上一定要设小，比如 512 或 128
    load_in_4bit=True,  # 必须开启
    device_map={"": 0},  # 核心操作：强制模型只准呆在 0 号显卡上
    local_files_only=True,
    # fix_mistral_regex=True,
)

# 3. 执行 GGUF 导出（确保你已经安装了 cmake）
model.save_pretrained_gguf(
    "qwen3.5_9b_q4_k_m",       # 导出的目标文件夹名
    tokenizer,
    quantization_method = "q4_k_m"  # 推荐的 4bit 量化方法
)

print("GGUF 转换完成！请检查 qwen3.5_9b_q4_k_m 文件夹。")

# 或者使用胰腺癌命令行 （亦可直接llama.cpp转换运行模型）
# cd /your/tb_disk/
# 如果 wget 429，尝试从其他源下载，或者直接复制这个脚本内容
# pip install gguf transformers -U

# 在终端运行
# python3 -m gguf.convert_hf_to_gguf /your/tb_disk/model_folder --outfile /your/tb_disk/qwen_final.gguf --outtype q4_k_m
