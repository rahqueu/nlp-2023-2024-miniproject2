# Define the two file paths to compare
file1_path = 'output/labels.txt'
file2_path = 'output/labels_nb.txt'

# Read the contents of both files
with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
    file1_lines = file1.readlines()
    file2_lines = file2.readlines()

# Compare the lines of the two files
diff_lines = []
line_number = 1  # Initialize line number

for line1, line2 in zip(file1_lines, file2_lines):
    if line1 != line2:
        diff_lines.append(f"Line {line_number}: File 1: {line1.strip()}, File 2: {line2.strip()}")
    line_number += 1

# Display the different lines
if len(diff_lines) > 0:
    print("Lines that are different:")
    for line in diff_lines:
        print(line)
else:
    print("The files are identical.")
    
print(len(diff_lines))