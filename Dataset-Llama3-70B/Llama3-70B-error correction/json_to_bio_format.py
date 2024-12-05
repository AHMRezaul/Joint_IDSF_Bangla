# This file is used to convert the JSON data files into the final dataset
# split into label, seq.in and seq.out file

import json
from tqdm import tqdm

# Assuming you have already loaded the JSON data into 'translated_data'
file_path_translated = 'slot-aligned_dataset/test-translated-aligned.json'  # Update this to your JSON file path

# Output file paths
label_file_path = 'Final_dataset/test/label'  # Update to your desired path
seq_in_file_path = 'Final_dataset/test/seq.in'  # Update to your desired path
seq_out_file_path = 'Final_dataset/test/seq.out'  # Update to your desired path

# Load the JSON data
with open(file_path_translated, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Open the output files for writing
with open(label_file_path, 'w', encoding='utf-8') as label_file, \
     open(seq_in_file_path, 'w', encoding='utf-8') as seq_in_file:
    for item in tqdm(data, desc="Splitting JSON"):
        # Write the intent label
        label_file.write(f"{item['intent']}\n")
        
        # Clean and write the input sequence (text)
        clean_text = item['text'].replace("<pad>", "").replace("</s>", "").strip()
        seq_in_file.write(f"{clean_text}\n")


def tokenize_and_tag(text, slots):
    tokens = [token for token in text.split() if token not in ["<pad>", "</s>"]]
    tags = ['O'] * len(tokens)  # Default all tags to 'O'

    for slot in slots:
        slot_value = slot['slotValue'].replace("<pad>", "").replace("</s>", "").strip()
        slot_tokens = slot_value.split()

        if len(slot_tokens) == 0:
            continue

        # Try to match individual slot tokens, not just the entire sequence
        for i in range(len(tokens)):
            match_count = 0
            for j in range(len(slot_tokens)):
                if i + j < len(tokens) and tokens[i + j] == slot_tokens[j]:
                    match_count += 1
                else:
                    break

            # If all tokens match, assign BIO tags
            if match_count == len(slot_tokens):
                tags[i] = f"B-{slot['slotName']}"
                if len(slot_tokens) > 1:
                    tags[i+1:i+len(slot_tokens)] = [f"I-{slot['slotName']}" for _ in slot_tokens[1:]]
                break
    
    return tags


with open(label_file_path, 'w', encoding='utf-8') as label_file, \
     open(seq_in_file_path, 'w', encoding='utf-8') as seq_in_file, \
     open(seq_out_file_path, 'w', encoding='utf-8') as seq_out_file:

    for item in tqdm(data, desc="Processing Data"):
        # Write the intent label
        label_file.write(f"{item['intent']}\n")
        
        # Clean and write the input sequence (text)
        clean_text = item['text'].replace("<pad>", "").replace("</s>", "").strip()
        seq_in_file.write(f"{clean_text}\n")
        
        # Generate BIO tags and write them to seq.out
        tags = tokenize_and_tag(item['text'], item['slots'])
        seq_out_file.write(' '.join(tags) + '\n')

print("Files have been generated in the original order:")
print(f"Label file: {label_file_path}")
print(f"Seq.in file: {seq_in_file_path}")
print(f"Seq.out file: {seq_out_file_path}")
