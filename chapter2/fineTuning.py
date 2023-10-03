from transformers import AutoModelForSequenceClassification
from transformers import AutoModel, AutoTokenizer
from transformers import Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

emotions = load_dataset("emotion")


model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

text = "this is a test"
inputs = tokenizer(text, return_tensors="pt")

num_labels = 6
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=num_labels
).to(device)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


batch_size = 64
logging_steps = len(emotions_encoded["train"])
model_name = f"{model_ckpt}-finetuned-emotion-tanmay"
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=True,
    log_level="error",
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=emotions_encoded["train"],
    eval_dataset=emotions_encoded["validation"],
    tokenizer=tokenizer,
)

trainer.train()

preds_output = trainer.predict(emotions_encoded["validation"])
y_preds = np.argmax(preds_output.predictions, axis=1)


# def extract_hidden_states(batch):
#     # Place model inputs on the GPU
#     inputs = {
#         k:v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names
#     }
#     # Extract last hidden states
#     with torch.no_grad():
#         last_hidden_state = model(**inputs).last_hidden_state
#     # Return vector for [CLS] token
#     return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}

# emotions_encoded.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
# emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)
# X_train = np.array(emotions_hidden['train']['hidden_state'])
# X_valid = np.array(emotions_hidden['validation']['hidden_state'])
# y_train = np.array(emotions_hidden['train']['label'])
# y_valid = np.array(emotions_hidden['validation']['label'])

labels = emotions["train"].features["label"].names
y_valid = np.array(emotions['validation']['label'])

def plot_confusion_matrix(y_preds, y_true, labels):
  print('*')*200
  cm = confusion_matrix(y_true, y_preds, normalize='true')
  fig, ax = plt.subplots(figsize=(6,6))
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
  disp.plot(cmap='Blues', values_format='.2f', ax=ax, colorbar=False)
  plt.title('Normalized confusion matrix')
  plt.show()

plot_confusion_matrix(y_preds, y_valid, labels)
