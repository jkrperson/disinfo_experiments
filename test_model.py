from models.classification_model import ClassifierModel
from data.verafiles import VeraFilesDataset
from transformers import AutoTokenizer
import torch.nn.functional as F

data = VeraFilesDataset("datasets/disinfo_unbalanced/test.csv", None)
model = ClassifierModel.load_from_checkpoint("fakenews_detection/mBERT_verafiles_unbalanced/version_0/checkpoints/best-checkpoint.ckpt")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")

text, label = data[13]

# 401 is correct

tokenized = tokenizer(text, padding=True, max_length=512, truncation=True, return_tensors="pt")
model.freeze()
preds = model(tokenized["input_ids"], tokenized["attention_mask"])
preds = F.softmax(preds, dim=1)

print(f"OUTPUTS: {preds}")
idx = preds.argmax(1)
print(f"TEXT: {text}")
print(f"ACTUAL LABEL: {data.id2label[label]} \t PREDICTED LABEL: {data.id2label[idx.item()]}")