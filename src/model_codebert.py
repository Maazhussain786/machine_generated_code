
import os
import json
import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

MODEL_NAME = "microsoft/codebert-base" 


def df_to_hfdataset(df: pd.DataFrame, tokenizer, max_length=512, code_col="code", token_batch_size: int = 4096):
    """
    Tokenize a pandas DataFrame into a HuggingFace Dataset with a visible tqdm progress bar.
    Tokenization is done in chunks (token_batch_size) to reduce memory peaks and show progress.
    """
    texts = df[code_col].astype(str).tolist()
    labels = df["label"].astype(int).tolist() if "label" in df.columns else None

    if len(texts) <= token_batch_size:
        print(f"🧪 Tokenizing {len(texts):,} samples (single batch)...")
        tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)
    else:
        print(f"🧪 Tokenizing {len(texts):,} samples in chunks of {token_batch_size} with progress bar...")
        input_ids = []
        attention_mask = []
        
        for i in tqdm(range(0, len(texts), token_batch_size), desc="Tokenizing"):
            chunk = texts[i:i + token_batch_size]
            enc = tokenizer(chunk, padding="max_length", truncation=True, max_length=max_length)
            input_ids.extend(enc["input_ids"])
            attention_mask.extend(enc["attention_mask"])
        tokenized = {"input_ids": input_ids, "attention_mask": attention_mask}

    ds = Dataset.from_dict(tokenized)
    if labels is not None:
        ds = ds.add_column("labels", labels)
    return ds


def compute_metrics(pred):
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "precision_macro": float(precision_score(labels, preds, average="macro")),
        "recall_macro": float(recall_score(labels, preds, average="macro")),
    }


def train_codebert(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    out_dir: str = "experiments/codebert_task_a",
    model_name: str = MODEL_NAME,
    epochs: int = 3,
    batch_size: int = 8,
    max_length: int = 256,
    lr: float = 2e-5,
    token_batch_size: int = 4096
):
    os.makedirs(out_dir, exist_ok=True)

    # Artifact check: if model exists, skip training
    model_path = os.path.join(out_dir, "pytorch_model.bin")
    artifact_flag = os.path.join(out_dir, "model_ready.joblib")
    if os.path.exists(model_path) and os.path.exists(artifact_flag):
        print(f"📦 Found existing CodeBERT model in {out_dir} — skipping training.")
        return joblib.load(artifact_flag)

    # Paths for tokenized cache
    cache_train = os.path.join(out_dir, "train_tokenized.joblib")
    cache_dev = os.path.join(out_dir, "dev_tokenized.joblib")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


    if os.path.exists(cache_train) and os.path.exists(cache_dev):
        print("📂 Loading cached tokenized data...")
        train_ds = joblib.load(cache_train)
        dev_ds = joblib.load(cache_dev)
    else:
        print("🧠 Tokenizing train and dev datasets (this may take a while)...")
        train_ds = df_to_hfdataset(train_df, tokenizer, max_length=max_length, code_col="code", token_batch_size=token_batch_size)
        dev_ds = df_to_hfdataset(dev_df, tokenizer, max_length=max_length, code_col="code", token_batch_size=token_batch_size)
        # Save tokenized datasets for future runs
        joblib.dump(train_ds, cache_train)
        joblib.dump(dev_ds, cache_dev)
        print("💾 Tokenized datasets cached.")


    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    training_args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(1, batch_size * 2),
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
        disable_tqdm=False,          
        logging_dir=os.path.join(out_dir, "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("🚀 Training CodeBERT...")
    trainer.train()

    print("📊 Evaluating on dev set...")
    eval_res = trainer.evaluate()
    preds = trainer.predict(dev_ds)
    pred_labels = np.argmax(preds.predictions, axis=1)


    pd.DataFrame({
        "y_true": dev_df["label"].astype(int).reset_index(drop=True),
        "y_pred": pred_labels
    }).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    filtered = {k: float(v) for k, v in eval_res.items() if isinstance(v, (int, float))}
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(filtered, fh, indent=2)


    joblib.dump(filtered, artifact_flag)
    print(f"✅ CodeBERT training done. Metrics saved to {out_dir}")

    return filtered