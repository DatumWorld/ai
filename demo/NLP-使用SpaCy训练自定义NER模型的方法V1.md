#  NLP-使用SpaCy训练自定义NER模型的方法V1
## 场景和问题：
识别业务文档中的的自定义NER（如中英文的各种业务术语实体）

## 总体流程的一个简单DEMO
spaCy v3 中，训练自定义命名实体识别（NER）模型不再推荐使用 Python 脚本循环训练，而是通过配置文件 (config.cfg) 结合命令行工具进行。 
0. 前提准备：
开发所需要的环境，含预训练模型（自行下载安装）、虚拟环境（python+CUDA）、训练数据集等，这里跳过此步

1. 数据标注与准备
<br>首先，你需要将文本标注为 spaCy 识别的格式。
<br>标注内容：
需要提供文本、实体在文中的起始位置（Start）、结束位置（End）以及标签名称（Label）。
<br>推荐工具：
<br>Prodigy：spaCy 官方开发的付费工具，与库集成度最高，支持主动学习。
<br>Label Studio：开源且功能强大，适合团队协作。
<br>格式转换：标注后的数据需转换为 .spacy 二进制格式。
```python
import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("zh")
doc_bin = DocBin()
# 你的训练集数据，处理好放在这里
data = [("这件合同包含异想模型服务", {"entities": [(6, 8, "MODEL_SERVICE")]})]
for text, annot in data:
    doc = nlp.make_doc(text)
    example = spacy.training.Example.from_dict(doc, annot)
    doc_bin.add(example.reference)
doc_bin.to_disk("./train.spacy")
```
同样的方法，创建验证集，.dev.spacy
 ```python
import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("zh")
doc_bin = DocBin()
# 你的验证集数据，处理好放在这里
data = [("新合同包含异想模型服务", {"entities": [(5, 7, "MODEL_SERVICE")]})]
for text, annot in data:
    doc = nlp.make_doc(text)
    example = spacy.training.Example.from_dict(doc, annot)
    doc_bin.add(example.reference)
doc_bin.to_disk("./dev.spacy")
```

2. 生成训练配置
<br>使用 spaCy Quickstart（https://spacy.io/usage/training） 工具生成基础配置文件。 

<br>如果是微调 zh_core_web_lg：在 Quickstart 中选择 Chinese，硬件选 CPU，Pipeline 选 ner。
<br>如果是微调 zh_core_web_trf：在 Quickstart 中选择 Chinese，硬件选 GPU (Transformer)，Pipeline 选 ner。 

<br>使用命令行初始化配置：
```bash
python -m spacy init fill-config base_config.cfg config.cfg
```

3. 以预训练模型为基础进行微调
<br>要基于 zh_core_web_lg 或 trf 进行微调，关键在于在 config.cfg 中设置初始化来源（Sourcing）。 

<br>修改 config.cfg：
<br>在 [initialize] 部分，指定要加载的预训练模型路径或名称：
```ini
[initialize]
vectors = "zh_core_web_lg"
```

保留原实体（可选）：
<br>如果你想保留模型原有的实体识别能力（如 PERSON, ORG），直接在已有模型上添加新标签进行训练；如果只想识别特定新类型，通常建议在空白模型上训练或冻结其他组件
<br>（通过配置文件[training]设置frozen_components = ["transformer"]和annotating_components = ["ner"]）。 

4. 执行训练
准备好 train.spacy、dev.spacy（验证集）和 config.cfg 后，放在同一个目录下，运行以下命令： 

```bash
python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./dev.spacy
```
如果是 Transformer 模型：确保已安装 spacy-transformers 并在有 GPU 的环境下运行，spaCy 会自动处理 Transformer 层的微调。 

5. 测试
比如使用下面代码测试
```python
import spacy
# 此时不需要加载 zh_core_web_trf，直接加载你练好的目录,best或者last模型
nlp = spacy.load("./output/model-last")

doc = nlp("新的一版合同包含异想模型服务")
for ent in doc.ents:
    print(ent.text, ent.label_,ent.start, ent.end,ent.has_vectot)
```
正常应该输出以下结果：

![img.png](img.png)

<br>建议：
<br>zh_core_web_lg：适合追求推理速度、标注样本量较小（几百条）的场景。
<br>zh_core_web_trf：适合追求极致准确率、有 GPU 算力支持且样本量相对充足的场景。