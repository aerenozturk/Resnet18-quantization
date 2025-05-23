import os
import re
import csv

FP32_FOLDER = "test_results_resnet18"
INT8_FOLDER = "test_results_resnet18_pt"
OUTPUT_CSV = "combined_results.csv"

CLASS_SCORE_REGEX = re.compile(r"Class:\s+(\d+),\s+Score:\s+([\d\.eE+-]+)")

def parse_log_file(file_path):
    class_score_pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = CLASS_SCORE_REGEX.search(line)
            if match:
                class_id = int(match.group(1))
                score = float(match.group(2))
                class_score_pairs.append((class_id, score))
    return class_score_pairs

def build_file_map(folder_path):
    file_map = {}
    for filename in os.listdir(folder_path):
        if filename.endswith("_log.txt"):
            stem = filename.replace("_log.txt", "")
            file_path = os.path.join(folder_path, filename)
            class_score_pairs = parse_log_file(file_path)
            file_map[stem] = class_score_pairs
    return file_map

def main():
    fp32_map = build_file_map(FP32_FOLDER)
    int8_map = build_file_map(INT8_FOLDER)

    all_keys = set(fp32_map.keys()).union(set(int8_map.keys()))
    all_keys = sorted(all_keys)

    fieldnames = [
        "image_id",
        "ground_truth",
        "class1_fp32", "score1_fp32",
        "class2_fp32", "score2_fp32",
        "class3_fp32", "score3_fp32",
        "class4_fp32", "score4_fp32",
        "class5_fp32", "score5_fp32",
        "class1_int8", "score1_int8",
        "class2_int8", "score2_int8",
        "class3_int8", "score3_int8",
        "class4_int8", "score4_int8",
        "class5_int8", "score5_int8",
    ]

    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for image_id in all_keys:
            row = {"image_id": image_id, "ground_truth": ""}

            fp32_preds = fp32_map.get(image_id, [])
            int8_preds = int8_map.get(image_id, [])

            for i in range(5):
                if i < len(fp32_preds):
                    row[f"class{i+1}_fp32"] = fp32_preds[i][0]
                    row[f"score{i+1}_fp32"] = fp32_preds[i][1]
                else:
                    row[f"class{i+1}_fp32"] = ""
                    row[f"score{i+1}_fp32"] = ""

                if i < len(int8_preds):
                    row[f"class{i+1}_int8"] = int8_preds[i][0]
                    row[f"score{i+1}_int8"] = int8_preds[i][1]
                else:
                    row[f"class{i+1}_int8"] = ""
                    row[f"score{i+1}_int8"] = ""

            writer.writerow(row)

    print(f"CSV file '{OUTPUT_CSV}' created successfully!")
    print("Remember to open it and manually fill in the 'ground_truth' column.")

if __name__ == "__main__":
    main()