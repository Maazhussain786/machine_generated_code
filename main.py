from pathlib import Path
import json
import time
import math

from src.data_loader import load_data
from src.preprocessing import preprocess_dataframe

import joblib
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # .../Machine_generated_code
RESULTS_DIR = BASE_DIR / "results"                  # .../Machine_generated_code/results
RESULTS_DIR.mkdir(exist_ok=True)

# 🔹 Your saved transformer (CodeBERT / DistilBERT) model directory
MODEL_C_DIR = RESULTS_DIR / "model_C_256"           # adjust if folder name different


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def build_hf_pipeline(model_dir: Path):
    """Load fine-tuned Model C and wrap in a HuggingFace pipeline."""

    # Decide device
    if torch.cuda.is_available():
        device_idx = 0
        device_name = torch.cuda.get_device_name(0)
        print(f"💻 Using GPU: {device_name}")
    else:
        device_idx = -1
        print("💻 No CUDA GPU detected — using CPU.")

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    id2label = getattr(model.config, "id2label", {0: "LABEL_0", 1: "LABEL_1"})
    label2id = {v: k for k, v in id2label.items()}

    def label_to_id(label_str: str) -> int:
        if label_str in label2id:
            return label2id[label_str]
        try:
            return int(label_str.split("_")[-1])
        except Exception:
            return 0

    clf = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device_idx,
        truncation=True,
        max_length=256,
        padding=True,
    )
    return clf, label_to_id


def predict_hf(clf, label_to_id, texts, batch_size: int = 32, label: str = "") -> np.ndarray:
    """
    Run HF pipeline and convert label strings to integer ids.
    Also prints a rough ETA based on batches processed so far.
    """
    texts = list(texts)
    total = len(texts)
    total_batches = math.ceil(total / batch_size)
    print(f"🧮 {label}: {total} samples, batch_size={batch_size} -> {total_batches} batches")

    preds = []
    start_time = time.time()

    for i in range(total_batches):
        batch_texts = texts[i * batch_size : (i + 1) * batch_size]
        out = clf(batch_texts, batch_size=batch_size)

        if isinstance(out, dict):
            preds.append(label_to_id(out["label"]))
        elif isinstance(out, list) and len(out) > 0:
            for o in out:
                preds.append(label_to_id(o["label"]))
        else:
            raise TypeError(f"Unexpected pipeline output type: {type(out)}")

        # Progress + ETA
        batches_done = i + 1
        elapsed = time.time() - start_time
        avg_per_batch = elapsed / batches_done
        remaining_batches = total_batches - batches_done
        eta_sec = remaining_batches * avg_per_batch

        mins = int(eta_sec // 60)
        secs = int(eta_sec % 60)
        print(
            f"   [{label}] Batch {batches_done}/{total_batches} "
            f"done — approx. remaining {mins:02d}:{secs:02d} (mm:ss)",
            end="\r",
        )

    print()  # newline after progress line
    return np.asarray(preds, dtype=int)


def eval_sklearn_model(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")
    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
    }


def eval_hf_model(clf, label_to_id, X, y, split_label: str):
    y_pred = predict_hf(clf, label_to_id, X, batch_size=32, label=split_label)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")
    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "y_pred": y_pred,
    }


# --------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------
def main():
    print("📂 Project root:", BASE_DIR)
    print("📂 Results dir :", RESULTS_DIR)

    # 1) Load & preprocess data
    print("\n📥 Loading data ...")
    train_df, val_df, test_df = load_data()

    print("🧹 Preprocessing data ...")
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
    test_df = preprocess_dataframe(test_df)

    X_val, y_val = val_df["code"], val_df["label"].astype(int).values
    X_test, y_test = test_df["code"], test_df["label"].astype(int).values

    # 2) Load Model C (your best / proposed model)
    print("\n============================================================")
    print("🤖 Model C (Proposed): Transformer from results/model_C_256")
    print("============================================================")

    if not MODEL_C_DIR.exists():
        raise FileNotFoundError(f"Model C directory not found at {MODEL_C_DIR}")

    clfC, label_to_id_C = build_hf_pipeline(MODEL_C_DIR)
    print(f"✅ Loaded Model C from {MODEL_C_DIR}")

    # 3) Load classical baselines A & B
    modelA_path = RESULTS_DIR / "model_A.pkl"
    modelB_path = RESULTS_DIR / "model_B.pkl"

    if not modelA_path.exists():
        raise FileNotFoundError(f"Model A not found at {modelA_path}")
    if not modelB_path.exists():
        raise FileNotFoundError(f"Model B not found at {modelB_path}")

    modelA = joblib.load(modelA_path)
    modelB = joblib.load(modelB_path)

    print(f"✅ Loaded Model A from {modelA_path}")
    print(f"✅ Loaded Model B from {modelB_path}")

    # 4) Evaluate on validation & test
    metrics = {"val": {}, "test": {}}

    print("\n🔹 Evaluating Model A (TF-IDF + LogReg)")
    metrics["val"]["A"] = eval_sklearn_model(modelA, X_val, y_val)
    metrics["test"]["A"] = eval_sklearn_model(modelA, X_test, y_test)

    print("\n🔹 Evaluating Model B (XGBoost + AST)")
    metrics["val"]["B"] = eval_sklearn_model(modelB, X_val, y_val)
    metrics["test"]["B"] = eval_sklearn_model(modelB, X_test, y_test)

    print("\n🔹 Evaluating Model C (Proposed Transformer)")
    val_stats_C = eval_hf_model(clfC, label_to_id_C, X_val, y_val, "Validation (Model C)")
    test_stats_C = eval_hf_model(clfC, label_to_id_C, X_test, y_test, "Test (Model C)")

    metrics["val"]["C"] = {k: v for k, v in val_stats_C.items() if k != "y_pred"}
    metrics["test"]["C"] = {k: v for k, v in test_stats_C.items() if k != "y_pred"}

    y_test_pred_C = test_stats_C["y_pred"]

    # 5) Print summary
    print("\n====================================")
    print("🏁 Baseline Comparison — Validation")
    print("====================================")
    for name, stats in metrics["val"].items():
        print(
            f"Model {name}: Acc = {stats['accuracy']:.4f}, "
            f"F1 = {stats['f1_weighted']:.4f}"
        )

    print("\n====================================")
    print("🏁 Baseline Comparison — Test")
    print("====================================")
    for name, stats in metrics["test"].items():
        print(
            f"Model {name}: Acc = {stats['accuracy']:.4f}, "
            f"F1 = {stats['f1_weighted']:.4f}"
        )

    print("\n📋 Classification report for Model C (Proposed) on Test set:")
    print(classification_report(y_test, y_test_pred_C, digits=4))

    # 6) Save metrics + predictions for Assignment-3 plots
    metrics_path = RESULTS_DIR / "baseline_comparison.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Saved metrics to {metrics_path}")

    np.save(RESULTS_DIR / "y_test.npy", y_test)
    np.save(RESULTS_DIR / "y_test_pred_C.npy", y_test_pred_C)
    print(f"💾 Saved y_test and y_test_pred_C to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
