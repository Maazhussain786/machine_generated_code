from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)


def load_metrics():
    metrics_path = RESULTS_DIR / "baseline_comparison.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Run main.py first to generate metrics."
        )
    with metrics_path.open("r") as f:
        metrics = json.load(f)
    return metrics


def plot_bar(metric_dict, metric_name: str, split: str, out_path: Path):
    labels = []
    values = []

    for model_name, stats in metric_dict.items():
        if stats is not None and metric_name in stats:
            labels.append(model_name)
            values.append(stats[metric_name])

    if not labels:
        print(f"⚠️ No values for {metric_name} on {split}, skipping.")
        return

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(f"{split} — {metric_name.replace('_', ' ').title()} Comparison")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    print(f"✅ Saved: {out_path}")


def plot_confusion_matrix(out_path: Path):
    y_true_path = RESULTS_DIR / "y_test.npy"
    y_pred_path = RESULTS_DIR / "y_test_pred_C.npy"  # predictions from proposed model

    if not y_true_path.exists() or not y_pred_path.exists():
        print("⚠️ Missing test predictions. Run main.py first.")
        return

    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)

    plt.figure()
    disp.plot(cmap="Blues", values_format="d", colorbar=False)
    plt.title("Confusion Matrix — Model C (Proposed) on Test Set")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    print(f"✅ Saved: {out_path}")


def main():
    print("📊 Loading metrics from:", RESULTS_DIR)
    metrics = load_metrics()

    # Validation plots
    plot_bar(metrics["val"], "accuracy", "Validation", PLOTS_DIR / "val_accuracy.pdf")
    plot_bar(metrics["val"], "f1_weighted", "Validation", PLOTS_DIR / "val_f1.pdf")

    # Test plots
    plot_bar(metrics["test"], "accuracy", "Test", PLOTS_DIR / "test_accuracy.pdf")
    plot_bar(metrics["test"], "f1_weighted", "Test", PLOTS_DIR / "test_f1.pdf")

    # Confusion matrix for Model C on Test
    plot_confusion_matrix(PLOTS_DIR / "confusion_matrix_C_test.pdf")

    print("\n🎉 All Assignment-3 plots generated successfully!")


if __name__ == "__main__":
    main()
