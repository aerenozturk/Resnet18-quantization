import csv

input_csv = "combined_results.csv"
output_csv = "updated_results.csv"

with open(input_csv, mode="r", encoding="utf-8") as infile, open(output_csv, mode="w", newline="", encoding="utf-8") as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames  
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    writer.writeheader()  

    for row in reader:
        # Check if 'class1_fp32' and 'class1_int8' are the same
        if row["class1_fp32"] == row["class1_int8"]:
            row["ground_truth"] = row["class1_fp32"]  # Assign the matched class
        else:
            row["ground_truth"] = ""  # Leave it blank

        writer.writerow(row)  # Write the updated row

print(f"Updated CSV saved as '{output_csv}'")
print("Remember to open it and manually fill in the 'ground_truth' column.")