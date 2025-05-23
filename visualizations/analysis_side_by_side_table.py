import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

CSV_FILE = "updated_results.csv"

def main():
    df = pd.read_csv(CSV_FILE)

    required_cols = [
        "ground_truth",
        "class1_fp32", "class2_fp32", "class3_fp32", "class4_fp32", "class5_fp32",
        "class1_int8", "class2_int8", "class3_int8", "class4_int8", "class5_int8"
    ]
    df = df.dropna(subset=required_cols)

    for col in required_cols:
        df[col] = df[col].astype(int)

    total = len(df)
    if total == 0:
        print("No valid rows found. Exiting.")
        return

    top1_correct_fp32 = (df["class1_fp32"] == df["ground_truth"]).sum()
    acc_fp32_top1 = top1_correct_fp32 / total

    top1_correct_int8 = (df["class1_int8"] == df["ground_truth"]).sum()
    acc_int8_top1 = top1_correct_int8 / total

    def is_in_top5_fp32(row):
        gt = row["ground_truth"]
        preds = [
            row["class1_fp32"],
            row["class2_fp32"],
            row["class3_fp32"],
            row["class4_fp32"],
            row["class5_fp32"]
        ]
        return int(gt in preds)

    df["correct_top5_fp32"] = df.apply(is_in_top5_fp32, axis=1)
    acc_fp32_top5 = df["correct_top5_fp32"].mean()

    def is_in_top5_int8(row):
        gt = row["ground_truth"]
        preds = [
            row["class1_int8"],
            row["class2_int8"],
            row["class3_int8"],
            row["class4_int8"],
            row["class5_int8"]
        ]
        return int(gt in preds)

    df["correct_top5_int8"] = df.apply(is_in_top5_int8, axis=1)
    acc_int8_top5 = df["correct_top5_int8"].mean()

    table_data = [
        ["FP32", f"{acc_fp32_top1*100:.2f}%", f"{acc_fp32_top5*100:.2f}%"],
        ["INT8", f"{acc_int8_top1*100:.2f}%", f"{acc_int8_top5*100:.2f}%"]
    ]
    print(f"Total Images Analyzed: {total}")
    print(tabulate(table_data, headers=["Model", "Top-1 Accuracy", "Top-5 Accuracy"], tablefmt="fancy_grid"))

    labels = ["FP32", "INT8"]
    top1_values = [acc_fp32_top1 * 100, acc_int8_top1 * 100]
    top5_values = [acc_fp32_top5 * 100, acc_int8_top5 * 100]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 4))
    bar1 = plt.bar([val - width/2 for val in x], top1_values, width=width, label="Top-1", color="blue")
    bar2 = plt.bar([val + width/2 for val in x], top5_values, width=width, label="Top-5", color="orange")

    plt.xticks(x, labels)
    plt.ylim([0, 100])
    plt.ylabel("Accuracy (%)")
    plt.title("Top-1 vs. Top-5 Accuracy by Model")
    plt.legend()

    for bars in [bar1, bar2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{height:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()
    plt.savefig("top1_top5_accuracy_comparison.png", dpi=150)
    plt.show()
    print("Bar chart saved to 'top1_top5_accuracy_comparison.png'.")

if __name__ == "__main__":
    main()