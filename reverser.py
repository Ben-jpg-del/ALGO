import csv

def reverse_csv_order(input_file, output_file):
    # Read the CSV file and store the rows.
    with open(input_file, mode='r', newline='') as infile:
        reader = csv.reader(infile)
        # Read the header separately.
        header = next(reader)
        # Read all remaining rows.
        rows = list(reader)
    
    # Reverse the order of the rows.
    rows.reverse()

    # Write the header and reversed rows to the output file.
    with open(output_file, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(rows)

if __name__ == "__main__":
    input_filename = "ONS.csv"   # Replace with your CSV file name
    output_filename = "ONS_1.csv" # Replace with your desired output file name
    reverse_csv_order(input_filename, output_filename)
    print(f"Reversed CSV written to {output_filename}")
