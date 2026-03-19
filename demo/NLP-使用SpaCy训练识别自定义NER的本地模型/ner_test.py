import spacy
# 此时不需要加载 zh_core_web_lg，直接加载你练好的目录
nlp = spacy.load("./output/model-best")

doc = nlp("新的一版合同包含异想模型服务")
for ent in doc.ents:
    print(ent.text, ent.label_,ent.start, ent.end)
