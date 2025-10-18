
import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

def compute_and_save_metrics(y_true, y_pred, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro"))
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.pdf"))
    plt.close()
    print("Saved metrics & confusion matrix to", out_dir)
    return metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, help="CSV with y_true,y_pred")
    ap.add_argument("--out_dir", default="plots")
    args = ap.parse_args()
    df = pd.read_csv(args.pred_csv)
    compute_and_save_metrics(df["y_true"].values, df["y_pred"].values, args.out_dir)
