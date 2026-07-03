import os
os.environ["HF_HOME"] = "./hf_cache"

from transformers import (
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import numpy as np
import torch

# Load tokenizer + model
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token

model_full = GPT2ForSequenceClassification.from_pretrained(
    "distilgpt2",
    num_labels=2,
    pad_token_id=tokenizer.eos_token_id
)

# Load SST-2
dataset = load_dataset("glue", "sst2")

def tokenize(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).mean()}

# Training args — kept minimal for speed
training_args = TrainingArguments(
    output_dir="./distilgpt2-sst2-full",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model_full,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()
model_full.save_pretrained("./distilgpt2-sst2-full")
tokenizer.save_pretrained("./distilgpt2-sst2-full")
print("Full fine-tuning done")