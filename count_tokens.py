import argparse
import multiprocessing
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import os

# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Count tokens using a tokenizer")
    parser.add_argument("model_name", type=str, help="Name of the model to use for tokenization")
    return parser.parse_args()

# Function to count tokens in a chunk of text with tqdm
def count_tokens(texts, tokenizer):
    total_tokens = 0
    for text in tqdm(texts, desc="Processing texts", position=0, leave=True):
        total_tokens += len(tokenizer.encode(text))
    return total_tokens

def save_to_csv(model_name, total_tokens, filename="token_count.csv"):
    # Create a DataFrame for the new data
    new_data = pd.DataFrame({
        "model_name": [model_name],
        "total_tokens": [total_tokens]
    })

    # If the file exists, append the data. Otherwise, create a new file.
    if os.path.exists(filename):
        existing_data = pd.read_csv(filename)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        updated_data.to_csv(filename, index=False)
    else:
        new_data.to_csv(filename, index=False)

def main():
    args = parse_args()
    model_name = args.model_name

    # Load dataset and tokenizer
    dataset = load_dataset("HAERAE-HUB/KOREAN-WEBTEXT")
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

    # Split dataset into chunks
    num_processes = multiprocessing.cpu_count()
    chunk_size = len(dataset['train']) // num_processes
    # chunk_size = len(dataset) // num_processes

    chunks = [dataset['train'][i * chunk_size:(i + 1) * chunk_size]['text'] for i in range(num_processes)]
    # chunks = [dataset[i * chunk_size:(i + 1) * chunk_size]['text'] for i in range(num_processes)]

    # Create a pool of processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        token_counts = pool.starmap(count_tokens, [(chunk, tokenizer) for chunk in chunks])

    # Sum the results from all processes
    total_tokens = sum(token_counts)

    print(f"Total tokens: {total_tokens}")

    # Save the model name and total tokens to the CSV file
    save_to_csv(model_name, total_tokens)

if __name__ == "__main__":
    main()
