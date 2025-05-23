import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

CSV_FILE = "updated_results.csv"
SAVE_FIGURES = True
SHOW_FIGURES = True

def finalize_figure(filename):
    if SAVE_FIGURES:
        plt.savefig(filename, dpi=200, bbox_inches='tight')
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()

df = pd.read_csv(CSV_FILE)

def get_top_k_preds(row, prefix="fp32", k=5):
    """Return a list of up to k predicted classes (ints) for a given row/model."""
    return [row[f"class{i}_{prefix}"] for i in range(1, k+1)]

def is_correct_topk(row, prefix="fp32", k=1):
    """
    Check if 'ground_truth' is in the top-k predictions for the specified model.
    k=1 => top-1; k=5 => top-5, etc.
    """
    top_k = get_top_k_preds(row, prefix=prefix, k=k)
    return row["ground_truth"] in top_k

def top1_confidence(row, prefix="fp32"):
    """Get the confidence (score) of the top-1 prediction."""
    return row[f"score1_{prefix}"]

def get_gt_rank_in_top5(row, prefix="fp32"):
    """
    Return the rank of the ground_truth in the top-5 predictions (1-based).
    If not in top-5, return 6.
    """
    top5 = get_top_k_preds(row, prefix=prefix, k=5)
    gt = row["ground_truth"]
    if gt in top5:
        return top5.index(gt) + 1
    else:
        return 6

df["correct_top1_fp32"] = df.apply(is_correct_topk, axis=1, prefix="fp32", k=1)
df["correct_top1_int8"] = df.apply(is_correct_topk, axis=1, prefix="int8", k=1)

df["correct_top5_fp32"] = df.apply(is_correct_topk, axis=1, prefix="fp32", k=5)
df["correct_top5_int8"] = df.apply(is_correct_topk, axis=1, prefix="int8", k=5)

top1_acc_fp32 = df["correct_top1_fp32"].mean()
top1_acc_int8 = df["correct_top1_int8"].mean()
top5_acc_fp32 = df["correct_top5_fp32"].mean()
top5_acc_int8 = df["correct_top5_int8"].mean()

print("==== Overall Accuracy ====")
print(f"FP32  - Top-1 Accuracy: {top1_acc_fp32:.4f}, Top-5 Accuracy: {top5_acc_fp32:.4f}")
print(f"INT8  - Top-1 Accuracy: {top1_acc_int8:.4f}, Top-5 Accuracy: {top5_acc_int8:.4f}")

fig, ax = plt.subplots(figsize=(6,4))
bar_labels = ["Top-1 FP32", "Top-1 INT8", "Top-5 FP32", "Top-5 INT8"]
bar_vals = [top1_acc_fp32, top1_acc_int8, top5_acc_fp32, top5_acc_int8]
colors = ["#1f77b4", "#ff7f0e", "#1f77b4", "#ff7f0e"]

ax.bar(bar_labels, bar_vals, color=colors)
ax.set_ylabel("Accuracy")
ax.set_title("Comparison of Top-1 and Top-5 Accuracy (FP32 vs. INT8)")
for i, v in enumerate(bar_vals):
    ax.text(i, v+0.001, f"{v:.2f}", ha='center', fontsize=8)
ax.set_ylim([0, 1])

plt.tight_layout()
finalize_figure("accuracy_bar_chart.png")

df["top1_score_fp32"] = df.apply(top1_confidence, axis=1, prefix="fp32")
df["top1_score_int8"] = df.apply(top1_confidence, axis=1, prefix="int8")

plt.figure(figsize=(7,5))
ax = sns.histplot(df["top1_score_fp32"], color="blue", label="FP32", kde=True, alpha=0.5, bins=30)
ax = sns.histplot(df["top1_score_int8"], color="orange", label="INT8", kde=True, alpha=0.5, bins=30)

handles, labels = ax.get_legend_handles_labels()
handles.append(Patch(facecolor='purple', alpha=0.3, label='Overlap region'))
ax.legend(handles=handles)

plt.title("Distribution of Top-1 Confidence Scores (FP32 vs. INT8)")
plt.xlabel("Confidence Score")
plt.ylabel("Count")

plt.tight_layout()
finalize_figure("top1_confidence_distribution.png")

df["score_diff"] = df["top1_score_fp32"] - df["top1_score_int8"]

plt.figure(figsize=(7,5))
score_diff_vals = df["score_diff"].values

sns.histplot(score_diff_vals, kde=True, bins=30, color="purple", palette="coolwarm")
plt.title("Histogram of (FP32 top-1 score) - (INT8 top-1 score)")
plt.xlabel("Score Difference (fp32 - int8)")
plt.ylabel("Count")

plt.tight_layout()
finalize_figure("top1_score_difference_hist.png")

same_top1_mask = (
    df["class1_fp32"] == df["class1_int8"]
)
num_same = same_top1_mask.sum()
num_diff = len(same_top1_mask) - num_same

plt.figure(figsize=(5,5))
colors = ["blue", "yellow"]
wedges, text, autotexts = plt.pie(
    [num_same, num_diff],
    labels=["Same Top-1", "Different Top-1"],
    autopct="%1.1f%%",
    colors=colors,
    startangle=140
)

plt.title("Agreement on Top-1 Prediction (FP32 vs. INT8)")

finalize_figure("top1_agreement_pie.png")

plt.figure(figsize=(6,6))
plt.scatter(df["top1_score_fp32"], df["top1_score_int8"], alpha=0.5)
plt.plot([0,1],[0,1], 'r--', lw=1)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel("FP32 Top-1 Score")
plt.ylabel("INT8 Top-1 Score")
plt.title("Scatter Plot of Top-1 Confidence: FP32 vs. INT8")

plt.tight_layout()
finalize_figure("top1_confidence_scatter.png")

df["rank_fp32"] = df.apply(get_gt_rank_in_top5, axis=1, prefix="fp32")
df["rank_int8"] = df.apply(get_gt_rank_in_top5, axis=1, prefix="int8")

ranks_fp32_counts = df["rank_fp32"].value_counts().sort_index()
ranks_int8_counts = df["rank_int8"].value_counts().sort_index()

all_ranks = [1,2,3,4,5,6]
fp32_values = [ranks_fp32_counts.get(r, 0) for r in all_ranks]
int8_values = [ranks_int8_counts.get(r, 0) for r in all_ranks]

x = np.arange(len(all_ranks))
width = 0.35

fig, ax = plt.subplots(figsize=(7,5))
rects1 = ax.bar(x - width/2, fp32_values, width, label="FP32")
rects2 = ax.bar(x + width/2, int8_values, width, label="INT8")

ax.set_xticks(x)
ax.set_xticklabels([str(r) for r in all_ranks])
ax.set_xlabel("Rank of Ground Truth (1=Top1, 6=Not in Top-5)")
ax.set_ylabel("Number of Images")
ax.set_title("Distribution of Ground Truth Ranks in Top-5 Predictions")
ax.legend()

for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f"{int(height)}",
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

plt.tight_layout()
finalize_figure("rank_distribution.png")

print("All figures have been generated and saved!")