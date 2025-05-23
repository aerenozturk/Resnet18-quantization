import csv

CSV_FILE = "updated_results.csv"  # Change this to your actual CSV path

def compute_top1_accuracy(csv_file):
    total = 0
    correct_fp32 = 0
    correct_int8 = 0

    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Skip any row that doesn't have a ground_truth value
            if not row["ground_truth"]:
                continue

            try:
                gt_class = int(row["ground_truth"])
                fp32_pred = int(row["class1_fp32"])
                int8_pred = int(row["class1_int8"])

                total += 1

                if fp32_pred == gt_class:
                    correct_fp32 += 1

                if int8_pred == gt_class:
                    correct_int8 += 1

            except ValueError:
                # If any row has non-integer or empty fields, skip it or handle as needed
                pass

    if total == 0:
        print("No valid rows found with ground_truth.")
        return

    top1_acc_fp32 = correct_fp32 / total * 100.0
    top1_acc_int8 = correct_int8 / total * 100.0

    print(f"Total images (with valid ground_truth): {total}")
    print(f"ResNet18 (FP32) Top-1 Accuracy: {top1_acc_fp32:.2f}%")
    print(f"ResNet18_pt (INT8) Top-1 Accuracy: {top1_acc_int8:.2f}%")

if __name__ == "__main__":
    compute_top1_accuracy(CSV_FILE)
