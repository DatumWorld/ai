import sys
import os

# 1. 彻底从 sys.modules 中阻断 cupy
# 如果已经加载了，强制删除
for mod in list(sys.modules.keys()):
    if mod.startswith("cupy"):
        del sys.modules[mod]


# 2. 伪造一个空的 cupy 模块，但这次要包含基础类型
class FakeNdarray: pass


class FakeCupy:
    ndarray = FakeNdarray

    def is_available(self): return False


# 注入伪造的模块，确保 isinstance 不报错
sys.modules["cupy"] = FakeCupy()

# 3. 屏蔽 GPU 环境
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import spacy
from spacy.cli.train import train


def run_train():
    # 确保没有尝试加载 gpu
    spacy.require_cpu()

    config_path = "config_lg.cfg"
    output_model = "./output"
    overrides = {
        "paths.train": "./train.spacy",
        "paths.dev": "./dev.spacy"
    }

    print("🚀 正在以纯 CPU/NumPy 模式启动训练...")
    try:
        # use_gpu=-1 强制关闭 GPU 探测
        train(config_path, output_model, overrides=overrides, use_gpu=-1)
    except Exception as e:
        print(f"❌ 训练中断: {e}")


if __name__ == "__main__":
    run_train()
