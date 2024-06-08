#!/bin/bash

# Array of model IDs
models=(
    # "42dot/42dot_LLM-SFT-1.3B"
    # "google/gemma-1.1-2b-it"
    # "yanolja/EEVE-Korean-Instruct-2.8B-v1.0"
    # "01-ai/Yi-6B-Chat"
    # "tiiuae/falcon-7b-instruct"
    # "mistralai/Mistral-7B-Instruct-v0.2"
    # "NousResearch/Meta-Llama-3-8B-Instruct"
    # "google/gemma-1.1-7b-it"
    # "upstage/SOLAR-10.7B-Instruct-v1.0"
    # "nlpai-lab/KULLM3"
    # "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
    # "KRAFTON/KORani-v1-13B"
    # "CohereForAI/aya-101"
    # "Qwen/Qwen1.5-MoE-A2.7B-Chat"
    # "OrionStarAI/Orion-14B-Chat"
    "01-ai/Yi-34B-Chat"
    "maywell/Yi-Ko-34B-Instruct"
    "CohereForAI/c4ai-command-r-v01"
    "tiiuae/falcon-40b-instruct"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "NousResearch/Meta-Llama-3-70B-Instruct"
    "CohereForAI/c4ai-command-r-plus"
    "databricks/dbrx-instruct"
    "mistralai/Mixtral-8x22B-Instruct-v0.1"
)

# Dataset, split, output directory, batch size, number of GPUs, and CSV filename
dataset="HAERAE-HUB/KOREAN-WEBTEXT"
split="train"
output_dir="./output"
batch_size=2
n_gpu=4
csv_filename="perplexity.csv"

# Loop through each model and run the command
for model_id in "${models[@]}"; do
    echo "Running model: $model_id"
    python ppl.py --model_id "$model_id" --dataset "$dataset" --split "$split" --output_dir "$output_dir" --batch_size "$batch_size" --n_gpu "$n_gpu" --csv_filename "$csv_filename"
done
