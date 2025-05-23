import pandas as pd
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
        print("No valid rows with ground_truth and top-5 predictions found. Exiting.")
        return

   
    def top5_correct_fp32(row):
        gt = row["ground_truth"]
        preds = [
            row["class1_fp32"],
            row["class2_fp32"],
            row["class3_fp32"],
            row["class4_fp32"],
            row["class5_fp32"]
        ]
        return 1 if gt in preds else 0

    # Similarly for INT8 model
    def top5_correct_int8(row):
        gt = row["ground_truth"]
        preds = [
            row["class1_int8"],
            row["class2_int8"],
            row["class3_int8"],
            row["class4_int8"],
            row["class5_int8"]
        ]
        return 1 if gt in preds else 0

    df["correct_top5_fp32"] = df.apply(top5_correct_fp32, axis=1)
    df["correct_top5_int8"] = df.apply(top5_correct_int8, axis=1)

    top5_acc_fp32 = df["correct_top5_fp32"].mean()  # fraction of correct
    top5_acc_int8 = df["correct_top5_int8"].mean()

    print(f"Total test images: {total}")
    print(f"ResNet18 (FP32) Top-5 Accuracy:  {top5_acc_fp32 * 100:.2f}%")
    print(f"ResNet18_pt (INT8) Top-5 Accuracy: {top5_acc_int8 * 100:.2f}%")


    plt.figure(figsize=(4, 4))
    models = ["FP32", "INT8"]
    accuracies = [top5_acc_fp32 * 100, top5_acc_int8 * 100]
    bar_colors = ["blue", "orange"]

    bars = plt.bar(models, accuracies, color=bar_colors)
    plt.ylim([0, 100])
    plt.ylabel("Top-5 Accuracy (%)")
    plt.title("ResNet18 Top-5 Accuracy Comparison")

    # Annotate bars with accuracy values
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig("top5_accuracy_comparison.png", dpi=150)
    plt.show()
    print("Saved bar chart as 'top5_accuracy_comparison.png'.")

if __name__ == "__main__":
    main()
