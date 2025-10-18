import os
import sys
import shutil
import joblib
import math
from glob import glob
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.features_ast import extract_features_df
from src.preprocess import basic_clean_code, add_length_features


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# XGBoost + TF-IDF bits
import xgboost as xgb
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer

def find_latest_checkpoint(folder):
    cands = glob(os.path.join(folder, "checkpoint-*"))
    if not cands:
        return None
    def idx(p):
        try:
            return int(os.path.basename(p).split("-")[-1])
        except:
            return 0
    cands = sorted(cands, key=idx, reverse=True)
    return cands[0]

def ensure_codebert_root(model_root):

    # quick check for model file
    for fname in ("pytorch_model.bin", "model.safetensors", "tf_model.h5", "flax_model.msgpack"):
        if os.path.exists(os.path.join(model_root, fname)):
            return True

    ck = find_latest_checkpoint(model_root)
    if not ck:
        return False

    cand_files = [
        "pytorch_model.bin", "model.safetensors", "config.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
        "tokenizer_config.json", "training_args.bin"
    ]
    copied = 0
    for f in cand_files:
        src = os.path.join(ck, f)
        dst = os.path.join(model_root, f)
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                shutil.copy(src, dst)
                copied += 1
            except Exception as e:
                print(f"⚠️ Failed copying {src} -> {dst}: {e}")
    return copied > 0

def predict_tfidf(test_df, model_dir):
    meta_path = os.path.join(model_dir, "hash_meta.joblib")
    model_path = os.path.join(model_dir, "sgd_model.joblib")
    if not os.path.exists(meta_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"TF-IDF artifacts not found in {model_dir}")
    meta = joblib.load(meta_path)
    clf = joblib.load(model_path)

    hv = HashingVectorizer(
        n_features=meta.get("n_features", 2**18),
        analyzer="word",
        ngram_range=(1, 3),
        alternate_sign=False,
        norm="l2",
    )

    df_proc = add_length_features(test_df.copy())
    texts = df_proc["code"].astype(str).map(basic_clean_code)
    Xw = hv.transform(texts.tolist())
    numeric_cols = ["char_count", "line_count", "avg_line_len", "comment_ratio"]
    Xnum = df_proc[numeric_cols].fillna(0.0).values
    X_test = hstack([Xw, csr_matrix(Xnum)])

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
    else:
        preds = clf.predict(X_test)
        probs = None
    return np.array(preds, dtype=int), (np.array(probs) if probs is not None else None)

def predict_xgboost(test_df, model_dir):
    X_test, _ = extract_features_df(test_df, code_col="code", lang_col="language")
    X_test = X_test.select_dtypes(exclude=["object"])
    saved_X_dev = os.path.join(model_dir, "X_dev_features.joblib")
    if os.path.exists(saved_X_dev):
        X_dev = joblib.load(saved_X_dev)
        X_test, _ = X_test.align(X_dev, join="right", axis=1, fill_value=0)
    X_test = X_test.fillna(0.0).astype(float)

    model_file = os.path.join(model_dir, "xgboost_model.json")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"XGBoost model file not found at {model_file}")
    booster = xgb.Booster()
    booster.load_model(model_file)
    dtest = xgb.DMatrix(X_test)
    probs = booster.predict(dtest)
    preds = (probs > 0.5).astype(int)
    return np.array(preds, dtype=int), np.array(probs)

def predict_codebert(test_df, model_dir, batch_size=64, max_length=256):
    ok = ensure_codebert_root(model_dir)
    if not ok and not os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        raise FileNotFoundError(f"No CodeBERT checkpoint/model found under {model_dir} and no checkpoint to copy.")

    # load tokenizer & model (prefer model_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    except Exception as e:
        # fallback: try to use checkpoint folder
        ck = find_latest_checkpoint(model_dir)
        if ck:
            tokenizer = AutoTokenizer.from_pretrained(ck, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(ck)
        else:
            raise RuntimeError(f"Failed to load CodeBERT from {model_dir}: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    texts = test_df["code"].astype(str).tolist()
    all_probs = []
    preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="CodeBERT inference"):
            batch_texts = texts[i:i+batch_size]
            tokens = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            outputs = model(**tokens)
            logits = outputs.logits.cpu().numpy()
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            prob1 = exp[:, 1] / exp.sum(axis=1)
            all_probs.extend(prob1.tolist())
            preds.extend(np.argmax(logits, axis=1).tolist())

    return np.array(preds, dtype=int), np.array(all_probs)

def evaluate_and_save(name, preds, probs, test_df, model_dir):
    y_true = test_df["label"].astype(int).values
    if probs is not None:
        preds_to_save = (probs > 0.5).astype(int)
    else:
        preds_to_save = np.array(preds, dtype=int)

    out_path = os.path.join(model_dir, "predictions_test.csv")
    pd.DataFrame({"y_true": y_true, "y_pred": preds_to_save}).to_csv(out_path, index=False)
    print(f"✅ Test predictions saved for {name} → {out_path}")

    print(f"\n--- Results for {name} ---")
    print("Counts y_true:", dict(pd.Series(y_true).value_counts()))
    print("Counts y_pred:", dict(pd.Series(preds_to_save).value_counts()))
    print(classification_report(y_true, preds_to_save, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, preds_to_save))
    acc = accuracy_score(y_true, preds_to_save)
    f1 = f1_score(y_true, preds_to_save, average="macro")
    return {"accuracy": float(acc), "f1_macro": float(f1)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default=os.path.join(ROOT, "data", "task_a", "task_a_test_set_sample.parquet"))
    parser.add_argument("--experiments_dir", default=os.path.join(ROOT, "experiments"))
    parser.add_argument("--models", nargs="+", default=["tfidf", "xgboost", "codebert"])
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    TEST_FILE = args.test_file
    EXPERIMENTS_DIR = args.experiments_dir
    MODEL_FOLDERS = {
        "tfidf": os.path.join(EXPERIMENTS_DIR, "tfidf_task_a"),
        "xgboost": os.path.join(EXPERIMENTS_DIR, "xgboost_task_a"),
        "codebert": os.path.join(EXPERIMENTS_DIR, "codebert_task_a"),
    }

    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Test file not found at {TEST_FILE}.")

    print("📥 Loading test set...")
    test_df = pd.read_parquet(TEST_FILE)
    print(f"✅ Loaded test set with {len(test_df):,} samples. Label counts:\n{test_df['label'].value_counts()}\n")

    results = {}

    if "tfidf" in args.models:
        try:
            print("\n" + "="*60)
            print("▶ TF-IDF evaluation")
            preds, probs = predict_tfidf(test_df, MODEL_FOLDERS["tfidf"])
            r = evaluate_and_save("tfidf", preds, probs, test_df, MODEL_FOLDERS["tfidf"])
            results["tfidf"] = r
        except Exception as e:
            print(f"⚠️ TF-IDF evaluation failed: {e}")

    if "xgboost" in args.models:
        try:
            print("\n" + "="*60)
            print("▶ XGBoost evaluation")
            preds, probs = predict_xgboost(test_df, MODEL_FOLDERS["xgboost"])
            r = evaluate_and_save("xgboost", preds, probs, test_df, MODEL_FOLDERS["xgboost"])
            results["xgboost"] = r
        except Exception as e:
            print(f"⚠️ XGBoost evaluation failed: {e}")

    if "codebert" in args.models:
        try:
            print("\n" + "="*60)
            print("▶ CodeBERT evaluation")
            preds, probs = predict_codebert(test_df, MODEL_FOLDERS["codebert"], batch_size=args.batch_size, max_length=args.max_length)
            r = evaluate_and_save("codebert", preds, probs, test_df, MODEL_FOLDERS["codebert"])
            results["codebert"] = r
        except Exception as e:
            print(f"⚠️ CodeBERT evaluation failed: {e}")

    print("\n" + "="*60)
    print("📊 Final Summary (test):")
    for m, v in results.items():
        print(f"  ▶ {m:8s} | Accuracy: {v['accuracy']:.4f} | F1_macro: {v['f1_macro']:.4f}")

if __name__ == "__main__":
    main()