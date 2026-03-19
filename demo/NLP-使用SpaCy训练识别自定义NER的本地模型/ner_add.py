import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("zh")
doc_bin = DocBin()
data = [("这件合同包含异想模型服务", {"entities": [(6, 8, "MODEL_SERVICE")]})]

for text, annot in data:
    doc = nlp.make_doc(text)
    example = spacy.training.Example.from_dict(doc, annot)
    doc_bin.add(example.reference)
doc_bin.to_disk("./train.spacy")