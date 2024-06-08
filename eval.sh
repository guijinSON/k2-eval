#!/bin/bash

# Usage: ./script.sh
# This script processes a predefined list of models.

# List of models
declare -a MODELS=(
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
    "CohereForAI/c4ai-command-r-v01"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "CohereForAI/c4ai-command-r-plus"
    "databricks/dbrx-instruct"
    "mistralai/Mixtral-8x22B-Instruct-v0.1"
)

# Iterate over each model in the list
for MODEL_PATH in "${MODELS[@]}"
do
    MODEL_NAME=$(echo $MODEL_PATH | tr '/' '_') 
    LOCAL_DIR="$MODEL_NAME"
    
    echo "Downloading Model: $MODEL_PATH to $LOCAL_DIR"
    python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_PATH', token='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',local_dir='$LOCAL_DIR')"

    echo "Running text generation for model: $MODEL_PATH"
    python text_generation.py --model_path $LOCAL_DIR

    echo "Evaluating model: $MODEL_PATH"
    python evaluate.py $LOCAL_DIR

    #Clear the Hugging Face cache directory
    echo "Clearing cache..."
    rm -rf $LOCAL_DIR
    rm -rf ~/.cache/huggingface
done