#!/bin/bash

python text_generation.py --model_path "42dot/42dot_LLM-SFT-1.3B"

python evaluate.py "42dot/42dot_LLM-SFT-1.3B"
