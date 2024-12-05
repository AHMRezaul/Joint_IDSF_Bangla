import json
import re

# Define file path (replace with your actual path)
file_path_translated = 'slot-aligned_dataset/dev-translated-aligned.json'  # Update this to your JSON file path

# Regular expression to match English characters (a-z, A-Z) and numbers (0-9)
english_char_num_regex = re.compile(r'[a-zA-Z0-9]')

def contains_english_char_or_number(text):
    # Search for any English character or number in the text
    return bool(re.search(english_char_num_regex, text))

# Load the JSON data
with open(file_path_translated, 'r', encoding='utf-8') as file:
    data = json.load(file)

# List to store indices of entries with English characters or numbers
english_char_num_indices = []

# Iterate over the data and check for English characters or numbers in the "text"
for index, item in enumerate(data):
    text = item.get('text', '')
    if contains_english_char_or_number(text):
        english_char_num_indices.append(index)

# Output the indices as a comma-separated list
if english_char_num_indices:
    print(f"Indices of entries with English characters or numbers: {', '.join(map(str, english_char_num_indices))}")
else:
    print("No entries contain English characters or numbers.")
