import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

CSV_FILE = "updated_results.csv"  

def main():
   
    df = pd.read_csv(CSV_FILE)
    
    df = df.dropna(subset=["ground_truth", "class1_fp32", "class1_int8"])
    
    df["ground_truth"] = df["ground_truth"].astype(int)
    df["class1_fp32"]  = df["class1_fp32"].astype(int)
    df["class1_int8"]  = df["class1_int8"].astype(int)
    

    total = len(df)
    if total == 0:
        print("No valid rows with ground_truth found. Exiting.")
        return

    df["correct_fp32"] = (df["class1_fp32"] == df["ground_truth"]).astype(int)
    df["correct_int8"] = (df["class1_int8"] == df["ground_truth"]).astype(int)

    acc_fp32 = df["correct_fp32"].mean()  # fraction correct
    acc_int8 = df["correct_int8"].mean()

    print(f"Total test images: {total}")
    print(f"ResNet18 (FP32) Top-1 Accuracy: {acc_fp32*100:.2f}%")
    print(f"ResNet18_pt (INT8) Top-1 Accuracy: {acc_int8*100:.2f}%")


    plt.figure(figsize=(4, 4))
    models = ["FP32", "INT8"]
    accuracies = [acc_fp32 * 100, acc_int8 * 100]
    bar_colors = ["blue", "orange"]

    bars = plt.bar(models, accuracies, color=bar_colors)
    plt.ylim([0, 100])
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("ResNet18 Top-1 Accuracy Comparison")

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
    plt.savefig("top1_accuracy_comparison.png", dpi=150)
    plt.close()
    print("Saved bar chart as 'top1_accuracy_comparison.png'.")

    
    y_true = df["ground_truth"].values
    y_pred_fp32 = df["class1_fp32"].values
    y_pred_int8 = df["class1_int8"].values

    cm_fp32 = confusion_matrix(y_true, y_pred_fp32, normalize='true')
    cm_int8 = confusion_matrix(y_true, y_pred_int8, normalize='true')


    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im1 = axes[0].imshow(cm_fp32, cmap="Blues", aspect="auto")
    axes[0].set_title("CM (Normalized): FP32")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(cm_int8, cmap="Greens", aspect="auto")
    axes[1].set_title("CM (Normalized): INT8")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

   

    plt.tight_layout()
    plt.savefig("confusion_matrices_normalized.png", dpi=150)
    plt.close()
    print("Saved normalized confusion matrix as 'confusion_matrices_normalized.png'.")

  
    df_fp32_wrong = df[df["correct_fp32"] == 0].copy()
    df_int8_wrong = df[df["correct_int8"] == 0].copy()

  
    pair_counts_fp32 = (
        df_fp32_wrong.groupby(["ground_truth", "class1_fp32"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    pair_counts_int8 = (
        df_int8_wrong.groupby(["ground_truth", "class1_int8"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    top10_fp32 = pair_counts_fp32.head(10)
    top10_int8 = pair_counts_int8.head(10)

    
    def plot_top_confusions(df_pairs, model_name, color):
        df_pairs["pair_label"] = df_pairs.apply(
            lambda row: f"{row['ground_truth']}->{row.iloc[1]}", axis=1
        )
        plt.figure(figsize=(6, 4))
        plt.barh(df_pairs["pair_label"], df_pairs["count"], color=color)
        plt.gca().invert_yaxis()  # so top pair is at top
        plt.xlabel("Count of Occurrences")
        plt.title(f"Top 10 Confusion Pairs - {model_name}")
        plt.tight_layout()
        plt.savefig(f"top10_confusion_pairs_{model_name.lower()}.png", dpi=150)
        plt.close()
        print(f"Saved top-10 confusion pairs for {model_name} as 'top10_confusion_pairs_{model_name.lower()}.png'.")

    plot_top_confusions(top10_fp32, "FP32", "skyblue")
    plot_top_confusions(top10_int8, "INT8", "lightgreen")

    print("All analysis and visualizations are complete.")

if __name__ == "__main__":
    main()
