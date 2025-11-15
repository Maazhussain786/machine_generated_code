# src/model_baseline_C.py
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import torch
import os
import pickle

def get_baseline_C(train_df, val_df, test_df):
    print("============================================================")
    print("🤖 Training baseline C (DistilBERT 256 tokens, dynamic padding) ...")

    model_dir = "results/model_C_256"

    # ✅ Skip retraining if model already exists
    if os.path.exists(model_dir):
        print(f"📦 Found existing model at {model_dir} — skipping training.")
        return None

    # ✅ Load fast tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Tokenization (no padding here → handled dynamically)
    def tokenize(batch):
        return tokenizer(batch["code"], truncation=True, max_length=256)

    # Create cache directory
    os.makedirs("cache", exist_ok=True)
    cache_train = "cache/train_ds_256.pkl"
    cache_val = "cache/val_ds_256.pkl"

    # ✅ Load tokenized datasets from cache (if available)
    if os.path.exists(cache_train) and os.path.exists(cache_val):
        print("📂 Loading tokenized datasets from cache...")
        with open(cache_train, "rb") as f:
            train_ds = pickle.load(f)
        with open(cache_val, "rb") as f:
            val_ds = pickle.load(f)
    else:
        print("🔄 Tokenizing datasets (first time only)...")
        train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
        val_ds = Dataset.from_pandas(val_df).map(tokenize, batched=True)
        with open(cache_train, "wb") as f:
            pickle.dump(train_ds, f)
        with open(cache_val, "wb") as f:
            pickle.dump(val_ds, f)

    # ✅ Remove unneeded columns
    if "code" in train_ds.column_names:
        train_ds = train_ds.remove_columns(["code"])
        val_ds = val_ds.remove_columns(["code"])

    # ✅ Dynamic padding — each batch padded to longest sample in that batch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ✅ Model setup
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    args = TrainingArguments(
        output_dir=model_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=8,     # larger batch possible (shorter seq)
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        gradient_accumulation_steps=1,
        fp16=True if torch.cuda.is_available() else False,
        save_total_limit=1,
        save_strategy="epoch",
        logging_dir='./logs',
        logging_strategy='epoch',
        load_best_model_at_end=True,
        report_to="none"
    )

    # ✅ Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,  # ← dynamic padding enabled here
    )

    # ✅ Train and Save
    print("🚀 Starting training...")
    trainer.train()
    trainer.save_model(model_dir)
    print(f"✅ Model saved to {model_dir}")

    return trainer