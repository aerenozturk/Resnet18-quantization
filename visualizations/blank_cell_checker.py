import csv

input_csv = "updated_results.csv"

with open(input_csv, mode="r", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    row_number = 1 

    empty_cells = [] 

    for row in reader:
        row_number += 1 

        for column in fieldnames:
            if row[column].strip() == "":  
                empty_cells.append((row_number, column))  

if empty_cells:
    print(f"Found {len(empty_cells)} blank cells:")
    for row_num, col_name in empty_cells:
        print(f" - Row {row_num}, Column '{col_name}' is empty")
else:
    print("No blank cells found in the CSV file!")
