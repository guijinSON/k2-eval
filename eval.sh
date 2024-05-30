#!/bin/bash

# Usage: ./script.sh [model_path]
# Example: ./script.sh "42dot/42dot_LLM-SFT-1.3B"

# Set the model path from the first command-line argument, defaulting to "42dot/42dot_LLM-SFT-1.3B" if not provided
MODEL_PATH=${1:-"42dot/42dot_LLM-SFT-1.3B"}

echo "Running text generation model..."
python text_generation.py --model_path "$MODEL_PATH"

echo "Evaluating the model..."
python evaluate.py "$MODEL_PATH"
