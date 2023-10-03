from transformers import AutoModel, AutoTokenizer
import torch

model_ckpt = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModel.from_pretrained(model_ckpt).to(device)

text = 'this is a test'
inputs = tokenizer(text, return_tensors='pt')
