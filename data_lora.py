import os
os.environ["HF_HOME"] = "./hf_cache"

from transformers import (
    GPT2ForSequenceClassification,
    GPT2TokenizerFast,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from peft import get_peft_model, LoraConfig, TaskType
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Fast (Rust-backed) tokenizer — much quicker than the slow GPT2Tokenizer over the full dataset
tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token

# Load SST-2
dataset = load_dataset("glue", "sst2")

def tokenize(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=128,
        # no fixed padding here — DataCollatorWithPadding pads per-batch to the
        # longest example in that batch instead of always padding to 128
    )

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

base_model = GPT2ForSequenceClassification.from_pretrained(
    "distilgpt2",
    num_labels=2,
    pad_token_id=tokenizer.eos_token_id
).to(device)

# Try multiple ranks — start with 8
for rank in [4, 8, 16, 32]:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["c_attn"],  # GPT-2 attention projection
        lora_dropout=0.1,
        bias="none",
    )
    lora_model = get_peft_model(deepcopy(base_model), peft_config)
    lora_model.print_trainable_parameters()

    training_args_lora = TrainingArguments(
        output_dir=f"./distilgpt2-sst2-lora-r{rank}",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        # 0, not >0 — on Windows, num_workers>0 spawns worker processes, which
        # crashes in a plain top-level script without an `if __name__ == "__main__"` guard
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer_lora = Trainer(
        model=lora_model,
        args=training_args_lora,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer_lora.train()

    # Merge LoRA weights into base model for clean activation collection
    merged = lora_model.merge_and_unload()
    merged.save_pretrained(f"./distilgpt2-sst2-lora-r{rank}-merged")
    print(f"LoRA rank {rank} done")
