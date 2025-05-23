import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

SAVE_FIGURES = True
SHOW_FIGURES = True

def finalize_figure(filename):
    """
    Saves and/or shows the current figure, depending on the flags at the top.
    """
    if SAVE_FIGURES:
        plt.savefig(filename, dpi=200, bbox_inches='tight')
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()

df = pd.read_csv("updated_results.csv")

def get_topk(row, prefix="fp32", k=5):
    """
    Returns a list of (class_id, score) for the top-k predictions from the specified model prefix.
    By default, k=5 for top-5.
    """
    pairs = []
    for i in range(1, k+1):
        cls_col = f"class{i}_{prefix}"
        score_col = f"score{i}_{prefix}"
        pairs.append((int(row[cls_col]), float(row[score_col])))
    return pairs

def get_topk_classes(row, prefix="fp32", k=5):
    """
    Returns the top-k class IDs as a set (or list) for the specified model prefix.
    """
    return [int(row[f"class{i}_{prefix}"]) for i in range(1, k+1)]

def get_topk_scores(row, prefix="fp32", k=5):
    """
    Returns the top-k scores as a list for the specified model prefix.
    """
    return [float(row[f"score{i}_{prefix}"]) for i in range(1, k+1)]

def rank_in_topk(row, prefix="fp32", k=5):
    """
    Returns the 1-based rank of the ground truth within top-k predictions of the given model.
    If not found in top-k, returns k+1 (which indicates "not found").
    """
    topk_classes = get_topk_classes(row, prefix=prefix, k=k)
    gt = int(row["ground_truth"])
    if gt in topk_classes:
        return topk_classes.index(gt) + 1
    else:
        return k + 1

intersection_sizes = []
jaccard_indices = []
for idx, row in df.iterrows():
    fp32_set = set(get_topk_classes(row, prefix="fp32", k=5))
    int8_set = set(get_topk_classes(row, prefix="int8", k=5))
    intersection = fp32_set.intersection(int8_set)
    union = fp32_set.union(int8_set)
    
    intersection_sizes.append(len(intersection))
    jaccard_indices.append(len(intersection)/len(union))

df["intersection_size_top5"] = intersection_sizes
df["jaccard_index_top5"] = jaccard_indices

def spearman_for_top5(row):
    fp32_top5 = get_topk(row, prefix="fp32", k=5)  # list of (class, score)
    int8_top5 = get_topk(row, prefix="int8", k=5)
    
    fp32_ranks = { cls_score[0]: i for i, cls_score in enumerate(fp32_top5) }  # 0-based rank
    int8_ranks = { cls_score[0]: i for i, cls_score in enumerate(int8_top5) }

    common_classes = set(fp32_ranks.keys()).intersection(set(int8_ranks.keys()))
    if len(common_classes) < 2:
        return np.nan
    
    fp32_rank_list = []
    int8_rank_list = []
    for c in common_classes:
        fp32_rank_list.append(fp32_ranks[c])
        int8_rank_list.append(int8_ranks[c])
    
    corr, _ = spearmanr(fp32_rank_list, int8_rank_list)
    return corr

df["spearman_top5"] = df.apply(spearman_for_top5, axis=1)

for r in range(1,6):
    df[f"rank{r}_score_diff"] = df[f"score{r}_fp32"] - df[f"score{r}_int8"]

plt.figure(figsize=(6,4))
sns.countplot(x="intersection_size_top5", data=df, palette="viridis")
plt.title("Distribution of Intersection Sizes (Top-5 Sets, FP32 vs. INT8)")
plt.xlabel("Number of Classes in Intersection")
plt.ylabel("Count of Images")

finalize_figure("adv_1_intersection_distribution.png")

plt.figure(figsize=(6,4))
sns.histplot(df["jaccard_index_top5"], kde=True, color="purple", bins=10)
plt.title("Distribution of Jaccard Index (Top-5 Sets)")
plt.xlabel("Jaccard Index (0=No Overlap, 1=Exact Match)")
plt.ylabel("Count")

finalize_figure("adv_2_jaccard_distribution.png")

plt.figure(figsize=(6,4))
sns.histplot(df["spearman_top5"], kde=True, color="orange", bins=10)
plt.title("Distribution of Spearman Correlation (Top-5 Intersection Ranks)")
plt.xlabel("Spearman Correlation")
plt.ylabel("Count")

finalize_figure("adv_3_spearman_correlation.png")

rank_diff_means = []
rank_diff_stds = []
for r in range(1,6):
    col = f"rank{r}_score_diff"
    rank_diff_means.append(df[col].mean())
    rank_diff_stds.append(df[col].std())

x = np.arange(1,6)
plt.figure(figsize=(7,5))
plt.errorbar(x, rank_diff_means, yerr=rank_diff_stds, fmt='-o', capsize=5, color="blue")
plt.title("Mean Score Difference by Rank (fp32 - int8)")
plt.xlabel("Rank (1=Top-1, 5=Top-5)")
plt.ylabel("Mean Difference in Score")
plt.axhline(0, color='red', linestyle='--', linewidth=1)

for i, mean_val in enumerate(rank_diff_means):
    plt.text(x[i], mean_val, f"{mean_val:.3f}", ha='center', va='bottom', fontsize=8)

finalize_figure("adv_4_score_diff_by_rank.png")

plt.figure(figsize=(7,5))
data_for_box = [df[f"rank{r}_score_diff"] for r in range(1,6)]
sns.boxplot(data=data_for_box)
plt.xticks(range(5), [f"Rank {r}" for r in range(1,6)])
plt.title("Boxplot of Score Differences by Rank (FP32 - INT8)")
plt.xlabel("Rank")
plt.ylabel("Score Difference")
plt.axhline(0, color='red', linestyle='--', linewidth=1)

finalize_figure("adv_5_boxplot_score_diff_by_rank.png")

def get_score_for_gt_if_in_top5(row, prefix="fp32"):
    """Returns the score of the ground truth class if in top-5, else np.nan."""
    top5 = get_topk(row, prefix=prefix, k=5)  # list of (class, score)
    gt = int(row["ground_truth"])
    for rankpos, (cls, sc) in enumerate(top5):
        if cls == gt:
            return sc
    return np.nan

df["gt_score_fp32"] = df.apply(get_score_for_gt_if_in_top5, axis=1, prefix="fp32")
df["gt_score_int8"] = df.apply(get_score_for_gt_if_in_top5, axis=1, prefix="int8")

mask_both_in_top5 = (~df["gt_score_fp32"].isna()) & (~df["gt_score_int8"].isna())
df_both_in_top5 = df[mask_both_in_top5].copy()
df_both_in_top5["gt_score_diff"] = df_both_in_top5["gt_score_fp32"] - df_both_in_top5["gt_score_int8"]

plt.figure(figsize=(6,4))
sns.violinplot(data=df_both_in_top5, y="gt_score_diff", color="green", inner="quartile")

plt.axvline(0, color='red', linestyle='--', linewidth=2)

plt.title("Distribution of Score Differences for Ground Truth Class\n(Only Where GT is in Top-5 for Both Models)")
plt.xlabel("Score Difference (fp32 - int8)")

finalize_figure("adv_6_violinplot_gt_score_diff.png")

median_val = df_both_in_top5["gt_score_diff"].median()
mean_val = df_both_in_top5["gt_score_diff"].mean()

plt.figure(figsize=(6,4))
sns.kdeplot(df_both_in_top5["gt_score_diff"], fill=True, color="darkgreen", alpha=0.7)

plt.axvline(median_val, color="black", linestyle="--", label=f"Median: {median_val:.3f}")
plt.axvline(mean_val, color="blue", linestyle=":", label=f"Mean: {mean_val:.3f}")
plt.axvline(0, color='red', linestyle="--", linewidth=2)

plt.title("Density Plot of Score Differences for Ground Truth Class\n(Only Where GT is in Top-5 for Both Models)")
plt.xlabel("Score Difference (fp32 - int8)")
plt.legend()

finalize_figure("adv_6_densityplot_gt_score_diff.png")

plt.figure(figsize=(6,4))
sns.kdeplot(df_both_in_top5["gt_score_diff"], fill=True, color="darkgreen", alpha=0.7)

plt.axvline(0, color='red', linestyle="--", linewidth=2)

plt.title("Density Plot of Score Differences for Ground Truth Class\n(Only Where GT is in Top-5 for Both Models)")
plt.xlabel("Score Difference (fp32 - int8)")

finalize_figure("adv_6_densityplot_gt_score_diff.png")

plt.figure(figsize=(6,4))
sns.histplot(df_both_in_top5["gt_score_diff"], bins=20, kde=True, color="darkgreen", alpha=0.6)

plt.axvline(0, color='red', linestyle="--", linewidth=2)

plt.title("Histogram & KDE of Score Differences for Ground Truth Class\n(Only Where GT is in Top-5 for Both Models)")
plt.xlabel("Score Difference (fp32 - int8)")

finalize_figure("adv_6_histogram_kde_gt_score_diff.png")

df["gt_rank_fp32"] = df.apply(rank_in_topk, axis=1, prefix="fp32", k=5)
df["gt_rank_int8"] = df.apply(rank_in_topk, axis=1, prefix="int8", k=5)

mask_both_in_top5 = (df["gt_rank_fp32"] <= 5) & (df["gt_rank_int8"] <= 5)
df_both_ranks = df[mask_both_in_top5].copy()
df_both_ranks["gt_rank_diff"] = df_both_ranks["gt_rank_fp32"] - df_both_ranks["gt_rank_int8"]

plt.figure(figsize=(6,4))
sns.histplot(df_both_ranks["gt_rank_diff"], kde=True, bins=20, color="royalblue", alpha=0.6)
plt.title("Distribution of (Rank_fp32 - Rank_int8) for Ground Truth\n(Only Where GT is in Both Top-5)")
plt.xlabel("Rank Difference (fp32 minus int8)")
plt.ylabel("Count")
plt.grid(axis="x", linestyle="--", alpha=0.5)

finalize_figure("adv_7_gt_rank_difference.png")

def score_overlap_metric(row):
    fp32_top5 = get_topk(row, prefix="fp32", k=5)  # [(class, score), ...]
    int8_top5 = get_topk(row, prefix="int8", k=5)

    fp32_dict = {cls_id: sc for cls_id, sc in fp32_top5}
    int8_dict = {cls_id: sc for cls_id, sc in int8_top5}
    common = set(fp32_dict.keys()).intersection(int8_dict.keys())

    overlap_sum = sum(min(fp32_dict[c], int8_dict[c]) for c in common)
    return overlap_sum

df["score_overlap"] = df.apply(score_overlap_metric, axis=1)

plt.figure(figsize=(8, 4))  # Increased figure width for better spacing

sns.histplot(df["score_overlap"], bins=20, kde=True, color="teal", alpha=0.6)

plt.axvspan(0.0, 0.3, color='red', alpha=0.1, label="Low Overlap")
plt.axvspan(0.3, 0.7, color='yellow', alpha=0.1, label="Moderate Overlap")
plt.axvspan(0.7, 1.0, color='green', alpha=0.1, label="High Overlap")

plt.axvline(0.0, color="red", linestyle="--", label="No Overlap")
plt.axvline(0.5, color="black", linestyle="--", label="Partial Overlap")
plt.axvline(1.0, color="green", linestyle="--", label="Perfect Overlap")

peak_x = df["score_overlap"].mode()[0]
peak_y = df["score_overlap"].value_counts().max()
plt.annotate(f"Peak: {peak_x:.2f}", xy=(peak_x, peak_y), xytext=(peak_x - 0.1, peak_y + 5),
             arrowprops=dict(arrowstyle="->"), fontsize=10, color="black")

plt.xticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
plt.grid(axis="x", linestyle="--", alpha=0.5)

plt.title("Distribution of Score Overlap Metric (Sum of min(FP32, INT8) for Common Classes)")
plt.xlabel("Score Overlap Metric")
plt.ylabel("Count")
plt.legend()  

finalize_figure("adv_8_score_overlap_distribution.png")

df["sum_top5_fp32"] = df.apply(lambda row: sum(get_topk_scores(row, prefix="fp32", k=5)), axis=1)
df["sum_top5_int8"] = df.apply(lambda row: sum(get_topk_scores(row, prefix="int8", k=5)), axis=1)

plt.figure(figsize=(6,4))
sns.boxplot(data=[df["sum_top5_fp32"], df["sum_top5_int8"]], palette=["blue","orange"])
plt.xticks([0,1], ["FP32", "INT8"])
plt.ylabel("Sum of Top-5 Confidence")
plt.title("Boxplot of Sum of Top-5 Confidence (FP32 vs. INT8)")
finalize_figure("adv_9_sum_top5_confidence_boxplot.png")

def spearman_score_correlation(row):
    fp32_top5 = get_topk(row, prefix="fp32", k=5)  # [(class, score), ...]
    int8_top5 = get_topk(row, prefix="int8", k=5)

    fp32_dict = {c: s for c, s in fp32_top5}
    int8_dict = {c: s for c, s in int8_top5}
    common = set(fp32_dict.keys()).intersection(int8_dict.keys())
    if len(common) < 2:
        return np.nan
    
    fp32_scores = []
    int8_scores = []
    for c in common:
        fp32_scores.append(fp32_dict[c])
        int8_scores.append(int8_dict[c])
    
    corr, _ = spearmanr(fp32_scores, int8_scores)
    return corr

df["spearman_score_top5"] = df.apply(spearman_score_correlation, axis=1)

plt.figure(figsize=(6,4))
sns.histplot(df["spearman_score_top5"], kde=True, color="teal", bins=10)
plt.title("Distribution of Spearman Correlation (Scores in Intersection)")
plt.xlabel("Spearman Correlation (scores)")
plt.ylabel("Count")

finalize_figure("adv_10_spearman_score_correlation.png")


print("Advanced visualizations complete. All figures saved!")