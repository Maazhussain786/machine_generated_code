# incremental TF-IDF-like training using HashingVectorizer + SGDClassifier
import os
import time
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, accuracy_score
from preprocess import basic_clean_code, add_length_features
from scipy.sparse import hstack, csr_matrix
import json

def train_tfidf(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    out_dir: str = "experiments/tfidf_run_incr",
    n_features: int = 2**18,
    batch_size: int = 1600,
    n_epochs: int = 5,
    random_state: int = 42,
    alpha: float = 1e-5,
    patience: int = 3  # early stopping patience
):
    """
    Incremental training using HashingVectorizer + SGDClassifier with partial_fit.
    - Early stopping on dev F1 score
    - Saves best model automatically
    - Regularization controlled by alpha
    """
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # ✨ Preprocess and add features
    train_df = add_length_features(train_df.copy())
    dev_df = add_length_features(dev_df.copy())

    train_texts = train_df["code"].astype(str).map(basic_clean_code)
    dev_texts = dev_df["code"].astype(str).map(basic_clean_code)
    y_dev = dev_df["label"].astype(int).values

    # 🧠 Vectorizer and classifier
    hv = HashingVectorizer(
        n_features=n_features,
        analyzer='word',
        ngram_range=(1, 3),
        alternate_sign=False,
        norm='l2'
    )

    clf = SGDClassifier(
        loss="log_loss",
        alpha=alpha,
        max_iter=1,
        tol=None,
        warm_start=True,
        random_state=random_state
    )
    classes = np.array([0, 1])

    n = len(train_df)
    chunk_size = batch_size
    steps_per_epoch = (n + chunk_size - 1) // chunk_size

    print(f"Incremental training: n={n}, chunk={chunk_size}, steps/epoch={steps_per_epoch}, n_epochs={n_epochs}")
    best_f1 = 0.0
    epochs_no_improve = 0


    Xw_dev = hv.transform(dev_texts.tolist())
    Xnum_dev = dev_df[["char_count", "line_count", "avg_line_len", "comment_ratio"]].fillna(0.0).values
    X_dev = hstack([Xw_dev, csr_matrix(Xnum_dev)])

    for epoch in range(n_epochs):
        print(f"\n🧠 Epoch {epoch+1}/{n_epochs}")
        idx = np.arange(n)
        if epoch > 0:
            np.random.shuffle(idx)

        for start in tqdm(range(0, n, chunk_size), desc=f"Epoch{epoch+1}"):
            batch_idx = idx[start:start+chunk_size]
            texts_batch = train_texts.iloc[batch_idx].tolist()
            y_batch = train_df["label"].iloc[batch_idx].astype(int).values

            X_batch = hv.transform(texts_batch)
            Xnum = train_df.iloc[batch_idx][["char_count", "line_count", "avg_line_len", "comment_ratio"]].fillna(0.0).values
            X_batch = hstack([X_batch, csr_matrix(Xnum)])

            clf.partial_fit(X_batch, y_batch, classes=classes)

        # 🧪 Evaluate on dev after each epoch
        y_pred_dev = clf.predict(X_dev)
        acc_now = accuracy_score(y_dev, y_pred_dev)
        f1_now = f1_score(y_dev, y_pred_dev, average="macro")
        print(f"📊 Dev Accuracy: {acc_now:.4f} | Dev F1: {f1_now:.4f}")

        # 💾 Save best model
        if f1_now > best_f1:
            best_f1 = f1_now
            epochs_no_improve = 0
            joblib.dump(clf, os.path.join(out_dir, "sgd_model_best.joblib"))
            joblib.dump({"n_features": n_features}, os.path.join(out_dir, "hash_meta.joblib"))
            print(f"✅ Best model updated (F1={best_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s)")

        # ⏸️ Early stopping
        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break

    # 🏁 Load best model for final eval
    clf = joblib.load(os.path.join(out_dir, "sgd_model_best.joblib"))
    y_pred = clf.predict(X_dev)
    metrics = {
        "accuracy": float(accuracy_score(y_dev, y_pred)),
        "f1_macro": float(f1_score(y_dev, y_pred, average="macro"))
    }

    # 📝 Save predictions and metrics
    pd.DataFrame({"y_true": y_dev, "y_pred": y_pred}).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\n✅ Training finished. Best Dev F1: {best_f1:.4f} | Total time: {(time.time()-t0)/60:.2f} min")
    print(f"📁 Results saved to: {out_dir}")

    return metrics