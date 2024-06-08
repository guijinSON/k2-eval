#!/bin/bash

# List of models
models=(
    "OrionStarAI/Orion-14B-Chat"
)

# Iterate over models and run the Python script
for model in "${models[@]}"; do
    echo "Processing model: $model"
    python count_tokens.py "$model"
done
