from datasets import list_datasets
from transformers import AutoTokenizer

# all_datasets = list_datasets()
# print(f'There are {len(all_datasets)} datasets in hub')


model_ckpt = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

encoded_text = tokenizer('Tokenizing text is a core task of NLP')
print(encoded_text)