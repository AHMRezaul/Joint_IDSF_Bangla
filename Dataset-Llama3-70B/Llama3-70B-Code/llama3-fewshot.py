import torch, accelerate
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import os, shutil
import json

def build_prompt(prompt_text, text_to_translate):
    prompt=f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{prompt_text}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{text_to_translate}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

if __name__ == "__main__":

    cache_dir = "/.cache" # replace with your cache directory for Llama Model

    now = datetime.now()
    current_datetime = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Current DateTime =", current_datetime)

# Few-shot prompt for translation and transliteration

    prompt_text = (
        "Translate only the values of 'text' and 'slotValue' from English to Bangla and transliterate only the Entity Names in the 'text'."
        " Translated words in 'slotValue' must exactly match the words in final 'text' including the suffixes."
        " Do not provide additional text apart from the mentioned task.\n\n"
        "Original English Dataset:\n"
        "[\n"
        "    {\n"
        "        \"text\": \"add sabrina salerno to the grime instrumentals playlist\",\n"
        "        \"intent\": \"AddToPlaylist\",\n"
        "        \"slots\": [\n"
        "            {\n"
        "                \"slotName\": \"artist\",\n"
        "                \"slotValue\": \"sabrina salerno\"\n"
        "            },\n"
        "            {\n"
        "                \"slotName\": \"playlist\",\n"
        "                \"slotValue\": \"grime instrumentals\"\n"
        "            }\n"
        "        ]\n"
        "    },\n"
        "    {\n"
        "        \"text\": \"i want to bring four people to a place that s close to downtown that serves churrascaria cuisine\",\n"
        "        \"intent\": \"BookRestaurant\",\n"
        "        \"slots\": [\n"
        "            {\n"
        "                \"slotName\": \"party_size_number\",\n"
        "                \"slotValue\": \"four\"\n"
        "            },\n"
        "            {\n"
        "                \"slotName\": \"spatial_relation\",\n"
        "                \"slotValue\": \"close\"\n"
        "            },\n"
        "            {\n"
        "                \"slotName\": \"poi\",\n"
        "                \"slotValue\": \"downtown\"\n"
        "            },\n"
        "            {\n"
        "                \"slotName\": \"restaurant_type\",\n"
        "                \"slotValue\": \"churrascaria\"\n"
        "            }\n"
        "        ]\n"
        "    },\n"
        "    {\n"
        "        \"text\": \"put lindsey cardinale into my hillary clinton s women s history month playlist\",\n"
        "        \"intent\": \"AddToPlaylist\",\n"
        "        \"slots\": [\n"
        "            {\n"
        "                \"slotName\": \"artist\",\n"
        "                \"slotValue\": \"lindsey cardinale\"\n"
        "            },\n"
        "            {\n"
        "                \"slotName\": \"playlist_owner\",\n"
        "                \"slotValue\": \"my\"\n"
        "            },\n"
        "            {\n"
        "                \"slotName\": \"playlist\",\n"
        "                \"slotValue\": \"hillary clinton s women s history month playlist\"\n"
        "            }\n"
        "        ]\n"
        "    }\n"
        "]\n\n"
        "Expected Bangla Dataset:\n"
        "[\n"
        "    {\n"
        "        \"text\": \"গ্রাইম ইন্সট্রুমেন্টাল প্লেলিস্টে সাবরিনা সালেরনোকে যোগ করুন\",\n"
        "        \"intent\": \"AddToPlaylist\",\n"
        "        \"slots\": [\n"
        "            {\n"
        "                \"slotName\": \"artist\",\n"
        "                \"slotValue\": \"সাবরিনা সালেরনোকে\"\n"
        "            },\n"
        "            {\n"
        "                \"slotName\": \"playlist\",\n"
        "                \"slotValue\": \"গ্রাইম ইন্সট্রুমেন্টাল\"\n"
        "            }\n"
        "        ]\n"
        "    },\n"
        "    {\n"
        "        \"text\": \"আমি চারজন লোককে এমন একটি স্থানে নিয়ে যেতে চাই যা ডাউনটাউনের কাছাকাছি এবং যেখানে চুরাস্কারিয়া রান্না পরিবেশন করা হয়\",\n"
        "        \"intent\": \"BookRestaurant\",\n"
        "        \"slots\": [\n"
        "            {\n"
        "                \"slotName\": \"party_size_number\",\n"
        "                \"slotValue\": \"চারজন\"\n"
        "            },\n"
        "            {\n"
        "                \"slotName\": \"spatial_relation\",\n"
        "                \"slotValue\": \"কাছাকাছি\"\n"
        "            },\n"
        "            {\n"
        "                \"slotName\": \"poi\",\n"
        "                \"slotValue\": \"ডাউনটাউনের\"\n"
        "            },\n"
        "            {\n"
        "                \"slotName\": \"restaurant_type\",\n"
        "                \"slotValue\": \"চুরাস্কারিয়া\"\n"
        "            }\n"
        "        ]\n"
        "    },\n"
        "    {\n"
        "        \"text\": \"আমার হিলারি ক্লিন্টনের নারী ইতিহাস মাস প্লেলিস্টে লিন্ডসে কার্ডিনালকে যোগ করুন\",\n"
        "        \"intent\": \"AddToPlaylist\",\n"
        "        \"slots\": [\n"
        "            {\n"
        "                \"slotName\": \"artist\",\n"
        "                \"slotValue\": \"লিন্ডসে কার্ডিনালকে\"\n"
        "            },\n"
        "            {\n"
        "                \"slotName\": \"playlist_owner\",\n"
        "                \"slotValue\": \"আমার\"\n"
        "            },\n"
        "            {\n"
        "                \"slotName\": \"playlist\",\n"
        "                \"slotValue\": \"হিলারি ক্লিন্টনের নারী ইতিহাস মাস প্লেলিস্টে\"\n"
        "            }\n"
        "        ]\n"
        "    }\n"
        "]"
    )
    
    def load_json(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def process_data(data):
        entries = []
        for entry in data:
            entry_string = json.dumps(entry)
            entries.append(entry_string)
        return entries


    file_path = "train.json"
    data = load_json(file_path)
    input_texts = process_data(data)


    # print(json_data)

    texts_to_translate = [build_prompt(prompt_text,text) for text in input_texts]
    
    model_path = "meta-llama/Meta-Llama-3-70B-Instruct"
    model_name = model_path.split("/")[-1]
    print(f"Model Name: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        token="your_token_here" # replace with your Llama token
        )
    tokenizer.pad_token = tokenizer.eos_token

    accelerator = Accelerator()
    device = accelerator.device
    print(f"accelerator device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token="your_token_here", # replace with your Llama token
        cache_dir=cache_dir,
        device_map="auto",
        offload_folder="offload",
        torch_dtype=torch.bfloat16
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_length": 2048
    }

    result_string = []

    result_string.append("[")
    for expert_text in tqdm(texts_to_translate):
        query_encoding = tokenizer.encode(expert_text)
        response_tensor = model.generate(
            torch.tensor(query_encoding).unsqueeze(dim=0).to(device),  
            **generation_kwargs
        ).squeeze()[len(query_encoding):]
        response = tokenizer.decode(response_tensor)

        
        print(f"{response},\n")
        result_string.append(f'"{response}"')
        result_string.append(",")

    if result_string[-1] == ",":
        result_string[-1] = "]"
    else:
        result_string.append("]")
    
    json_string = ''.join(result_string)

    result = json.loads(json_string)

    save_path = 'path_to_save.json' # replace with your save path

    with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, indent=4)

    print(f"\nCompleted and saved as JSON")