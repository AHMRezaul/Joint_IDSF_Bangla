import json
from fuzzywuzzy import fuzz

# Define file paths (replace with actual paths)
input_json_path = "original_dataset/test-translated.json"
output_json_path = "slot-aligned_dataset/test-translated-aligned.json"

def align_slot_values(translated_text, translated_slots):
    for slot in translated_slots:
        words = translated_text.split()  # Split the text into words
        slot_words = slot['slotValue'].split()  # Split the slot value into words
        
        # Handle multiple word slot values by focusing on the final word
        matches = slot_words[:-1]  # Keep all words before the final one unchanged
        last_slot_word = slot_words[-1]  # Focus on the final word
        
        # Fuzzy match for the final word in the text
        best_match, best_score = None, 0
        for word in words:
            score = fuzz.partial_ratio(last_slot_word.lower(), word.lower())
            if score > best_score:
                best_score = score
                best_match = word

        # If a good match is found, replace the final word of the slot value
        if best_score >= 50:
            matches.append(best_match)  # Append the best match for the final word
            slot['slotValue'] = ' '.join(matches)  # Update the slot value

def process_dataset(input_path, output_path):
    # Load the dataset from the input JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Loop through each entry in the dataset and align slot values
    for entry in dataset:
        translated_text = entry['text']  # Extract the translated text
        translated_slots = entry['slots']  # Extract the slots
        
        # Align slot values for the current entry
        align_slot_values(translated_text, translated_slots)

    # Write the updated dataset to the output JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    process_dataset(input_json_path, output_json_path)
    print(f"Alignment completed. Updated dataset saved to {output_json_path}")
