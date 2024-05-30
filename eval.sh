#!/bin/bash

# Usage: ./script.sh
# This script processes a predefined list of models.

# List of models
declare -a MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "CohereForAI/c4ai-command-r-plus"
)


# Iterate over each model in the list
for MODEL_PATH in "${MODELS[@]}"
do
    echo "Downloading Model: $MODEL_PATH"
    python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_PATH', token='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',local_dir='eval_model/')"

    echo "Running text generation for model: $MODEL_PATH"
    python text_generation.py --model_path eval_model

    echo "Evaluating model: $MODEL_PATH"
    python evaluate.py eval_model

    # Clear the Hugging Face cache directory
    echo "Clearing cache..."
    rm -rf eval_model
    rm -rf ~/.cache/huggingface
done
