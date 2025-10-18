
import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
)

plt.rcParams.update({
    "figure.figsize": (8, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

MODEL_NAMES = ["tfidf", "xgboost", "codebert"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_preds_from_experiment(exp_folder):
    """
    Try to load predictions file and probability file (if exists).
    Return (y_true, y_pred, y_proba_or_none).
    Accepted names for probability files: preds_proba.joblib, preds_proba.csv, probs.joblib
    """
    preds_csv = None
    for fname in ("predictions_test.csv", "predictions.csv"):
        p = os.path.join(exp_folder, fname)
        if os.path.exists(p):
            preds_csv = p
            break
    if preds_csv is None:
        raise FileNotFoundError(f"No predictions file found in {exp_folder}")

    df = pd.read_csv(preds_csv)
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError(f"Predictions file {preds_csv} missing 'y_true'/'y_pred' columns")

    y_true = df["y_true"].astype(int).values
    y_pred = df["y_pred"].astype(int).values

    proba = None
    for pf in ("preds_proba.joblib", "pred_probs.joblib", "probs.joblib", "preds_proba.csv", "pred_probs.csv"):
        ppath = os.path.join(exp_folder, pf)
        if os.path.exists(ppath):
            try:
                if ppath.endswith(".joblib"):
                    proba = joblib.load(ppath)
                else:
                    proba = pd.read_csv(ppath).values.squeeze()
               
                proba = np.asarray(proba)
                break
            except Exception:
                proba = None

    if proba is None and "y_proba" in df.columns:
        proba = df["y_proba"].astype(float).values
    if proba is None and "probs" in df.columns:
        proba = df["probs"].astype(float).values

    return y_true, y_pred, proba, preds_csv


def plot_confusion_matrix(y_true, y_pred, out_path, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([0, 1])
    ax.set_yticklabels([0, 1])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path + ".png", bbox_inches="tight", dpi=150)
    fig.savefig(out_path + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(y_true, y_score, out_base, title_prefix=""):
    """
    y_score: probability for positive class (shape (n,))
    """
    if y_score is None:
        return None
 
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title_prefix} ROC curve")
    ax.legend(loc="lower right")
    fig.savefig(out_base + "_roc.png", bbox_inches="tight", dpi=150)
    fig.savefig(out_base + "_roc.pdf", bbox_inches="tight")
    plt.close(fig)

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, lw=2, label=f"PR AUC = {pr_auc:.4f}")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title_prefix} Precision-Recall")
    ax.legend(loc="lower left")
    fig.savefig(out_base + "_pr.png", bbox_inches="tight", dpi=150)
    fig.savefig(out_base + "_pr.pdf", bbox_inches="tight")
    plt.close(fig)

    return {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)}


def pretty_print_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))


def per_model_plots(model_name, exp_folder, plots_dir, task):
    plot_dir = ensure_dir(os.path.join(plots_dir, f"{model_name}_{task}"))
    try:
        y_true, y_pred, y_proba, pred_file = load_preds_from_experiment(exp_folder)
    except Exception as e:
        raise RuntimeError(f"Failed to load predictions for {model_name} in {exp_folder}: {e}")

    # metrics
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    metrics = {"accuracy": float(acc), "f1_macro": float(f1m)}
    print(f"\nModel: {model_name} — Acc: {acc:.4f} | F1_macro: {f1m:.4f}")
    pretty_print_report(y_true, y_pred)

    try:
        with open(os.path.join(exp_folder, "metrics_test.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)
    except Exception:
        pass

    # Confusion matrix plot
    cm_path = os.path.join(plot_dir, "confusion_matrix")
    plot_confusion_matrix(y_true, y_pred, cm_path, title=f"{model_name} confusion matrix")


    if y_proba is not None:
        rocpr = plot_roc_pr(y_true, y_proba, os.path.join(plot_dir, f"{model_name}_scores"),
                            title_prefix=model_name)
        if rocpr:
            metrics.update(rocpr)


    out_summary = os.path.join(plot_dir, "summary_metrics.csv")
    pd.DataFrame([metrics]).to_csv(out_summary, index=False)
    return metrics


def combined_comparison_plot(results_dict, plots_dir, task):
    """
    results_dict: {model: {"accuracy":.., "f1_macro":.., ...}}
    Will make a grouped bar chart of Accuracy and F1_macro (and ROC AUC if present).
    """
    plot_dir = ensure_dir(plots_dir)
    models = list(results_dict.keys())
    accs = [results_dict[m].get("accuracy", np.nan) for m in models]
    f1s = [results_dict[m].get("f1_macro", np.nan) for m in models]
    roc = [results_dict[m].get("roc_auc", np.nan) for m in models]

    x = np.arange(len(models))
    width = 0.28

    fig, ax = plt.subplots()
    ax.bar(x - width/2, accs, width, label="Accuracy")
    ax.bar(x + width/2, f1s, width, label="F1 (macro)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Model comparison ({task})")
    ax.legend()
    for i, v in enumerate(accs):
        ax.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    for i, v in enumerate(f1s):
        ax.text(i + width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    out_base = os.path.join(plot_dir, f"comparison_{task}")
    fig.savefig(out_base + ".png", bbox_inches="tight", dpi=150)
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)

    if not np.all(np.isnan(roc)):
        fig, ax = plt.subplots()
        ax.bar(models, roc)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"ROC AUC comparison ({task})")
        for i, v in enumerate(roc):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        out_base = os.path.join(plot_dir, f"comparison_roc_{task}")
        fig.savefig(out_base + ".png", bbox_inches="tight", dpi=150)
        fig.savefig(out_base + ".pdf", bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="task_a")
    parser.add_argument("--experiments", default="experiments")
    parser.add_argument("--plots_dir", default="plots")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES)
    parser.add_argument("--test_file", default=None, help="optional test file (for info only)")
    args = parser.parse_args()

    if args.test_file and os.path.exists(args.test_file):
        df_test = pd.read_parquet(args.test_file)
        print("Test set loaded:", args.test_file)
        print(df_test["label"].value_counts(), "\n")

    results_all = {}
    for model_name in args.models:
        exp_folder = os.path.join(args.experiments, f"{model_name}_{args.task}")
        if not os.path.exists(exp_folder):
            print(f"⚠️ Experiment folder not found: {exp_folder} — skipping {model_name}")
            continue
        try:
            metrics = per_model_plots(model_name, exp_folder, args.plots_dir, args.task)
            results_all[model_name] = metrics
        except Exception as e:
            print(f"⚠️ Failed processing {model_name}: {e}")

    if results_all:
        combined_comparison_plot(results_all, args.plots_dir, args.task)
        summary_rows = []
        for m, v in results_all.items():
            r = {"model": m}
            r.update(v)
            summary_rows.append(r)
        pd.DataFrame(summary_rows).to_csv(os.path.join(args.plots_dir, f"summary_{args.task}.csv"), index=False)
        print(f"\n✅ all plots saved to {os.path.abspath(args.plots_dir)}")
    else:
        print("No results were generated (no models processed).")


if __name__ == "__main__":
    main()