import re
import csv

# Input file
input_file = 'sorted_results_for_grid_search.txt'  # Text file with multiple lines
output_file = 'grid_search.csv'

# Extract variables from each line in the file
def extract_variables(line):
    pattern = r'END (.+?)(?= total time|$)'
    match = re.search(pattern, line)
    if match:
        variables = match.group(1)
        pairs = [item.strip() for item in variables.split(',') if '=' in item]
        unique_pairs = list(dict.fromkeys(pairs))
        data = {}
        for pair in unique_pairs:
            key, value = pair.split('=')
            data[key.strip()] = value.strip()
        return data
    return None

# Read lines from the input file and extract variables
all_data = []
columns = set()
with open(input_file, 'r') as file:
    for line in file:
        extracted = extract_variables(line)
        if extracted:
            all_data.append(extracted)
            columns.update(extracted.keys())

# Write the extracted data to a CSV file
if all_data:
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(columns))
        writer.writeheader()
        for row in all_data:
            writer.writerow(row)
    print(f"Variables successfully extracted and written to '{output_file}'.")
else:
    print("No variables found in the input file.")

