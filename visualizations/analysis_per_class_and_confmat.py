import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tabulate import tabulate

CSV_FILE = "updated_results.csv"

def main():

    df = pd.read_csv(CSV_FILE)

    required_cols = ["ground_truth", "class1_fp32", "class1_int8"]
    df = df.dropna(subset=required_cols)

    for col in required_cols:
        df[col] = df[col].astype(int)

    total = len(df)
    if total == 0:
        print("No valid rows with ground_truth, class1_fp32, class1_int8. Exiting.")
        return

    y_true = df["ground_truth"].values
    y_pred_fp32 = df["class1_fp32"].values
    y_pred_int8 = df["class1_int8"].values

    unique_classes = sorted(df["ground_truth"].unique())

    stats = {}
    for c in unique_classes:
        stats[c] = {"count": 0, "correct_fp32": 0, "correct_int8": 0}

    for gt, fp32_pred, int8_pred in zip(y_true, y_pred_fp32, y_pred_int8):
        stats[gt]["count"] += 1
        if fp32_pred == gt:
            stats[gt]["correct_fp32"] += 1
        if int8_pred == gt:
            stats[gt]["correct_int8"] += 1

    per_class_accuracy_fp32 = []
    per_class_accuracy_int8 = []
    class_labels = []

    for c in unique_classes:
        count_c = stats[c]["count"]
        if count_c > 0:
            acc_fp32_c = stats[c]["correct_fp32"] / count_c
            acc_int8_c = stats[c]["correct_int8"] / count_c
        else:
            acc_fp32_c = 0.0
            acc_int8_c = 0.0

        per_class_accuracy_fp32.append(acc_fp32_c * 100.0)
        per_class_accuracy_int8.append(acc_int8_c * 100.0)
        class_labels.append(c)

    x = np.arange(len(class_labels))
    width = 0.4

    plt.figure(figsize=(max(8, len(class_labels)*0.4), 5))

    plt.bar(x - width/2, per_class_accuracy_fp32, width, label="FP32", color="blue")
    plt.bar(x + width/2, per_class_accuracy_int8, width, label="INT8", color="orange")

    plt.xticks(x, class_labels, rotation=45, ha="right")
    plt.ylim([0, 100])
    plt.ylabel("Per-Class Accuracy (%)")
    plt.title("Per-Class Top-1 Accuracy Comparison")
    plt.legend()

    plt.tight_layout()
    plt.savefig("per_class_accuracy_comparison.png", dpi=150)
    plt.show()
    print("Saved per-class accuracy chart as 'per_class_accuracy_comparison.png'.")

    table_data = []
    for i, c in enumerate(class_labels):
        table_data.append([
            c,
            f"{per_class_accuracy_fp32[i]:.2f}%",
            f"{per_class_accuracy_int8[i]:.2f}%"
        ])

    print("\nPer-Class Accuracy Table:\n")
    print(tabulate(table_data, headers=["Class", "Acc (FP32)", "Acc (INT8)"], tablefmt="fancy_grid"))

    from sklearn.metrics import ConfusionMatrixDisplay

    cm_fp32 = confusion_matrix(y_true, y_pred_fp32, labels=unique_classes, normalize='true')
    cm_int8 = confusion_matrix(y_true, y_pred_int8, labels=unique_classes, normalize='true')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    disp_fp32 = ConfusionMatrixDisplay(cm_fp32, display_labels=unique_classes)
    disp_fp32.plot(ax=axes[0], cmap=plt.cm.Blues, colorbar=False)
    axes[0].set_title("Confusion Matrix (Normalized) - FP32")
    axes[0].set_xlabel("Predicted Class")
    axes[0].set_ylabel("True Class")

    disp_int8 = ConfusionMatrixDisplay(cm_int8, display_labels=unique_classes)
    disp_int8.plot(ax=axes[1], cmap=plt.cm.Oranges, colorbar=False)
    axes[1].set_title("Confusion Matrix (Normalized) - INT8")
    axes[1].set_xlabel("Predicted Class")
    axes[1].set_ylabel("True Class")

    fig.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("confusion_matrices_fp32_int8.png", dpi=150)
    plt.show()
    print("Saved normalized confusion matrices as 'confusion_matrices_fp32_int8.png'.")

    print("\nAnalysis complete. Per-class accuracy and confusion matrix visualizations saved.")

if __name__ == "__main__":
    main()