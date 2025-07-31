import os

# Path to your train/val label files (replace with actual path)
label_dir = r'c:\Users\ADMIN\Desktop\drive\yellow\val\labels'

# The incorrect class IDs to be replaced
incorrect_ids = {'0', '1', '2'}
correct_id = '3'

for filename in os.listdir(label_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(label_dir, filename)
        
        with open(filepath, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts and parts[0] in incorrect_ids:
                parts[0] = correct_id
            new_lines.append(' '.join(parts) + '\n')

        with open(filepath, 'w') as f:
            f.writelines(new_lines)

print("✅ Label correction complete.")
