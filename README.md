# k2-eval

### Cloning the Repository
Clone the repository to your local machine using:
```
git clone https://github.com/guijinSON/k2-eval.git
```

### Dependencies
Install all required dependencies by running the following command in your terminal:

```python
pip install vllm
pip install outlines==0.0.39
pip install kiwipiepy
pip install jamo
pip install pandas
pip install datasets
pip install transformers
```

#### Running the Script

```bash
./script.sh [model_path]
```

**Parameters:**

- `model_path` (optional): The path to the model you want to use. Defaults to `"42dot/42dot_LLM-SFT-1.3B"` if not specified.

**Examples:**

1. Using the default model path:
   ```bash
   ./script.sh
   ```
2. Specifying a custom model path:
   ```bash
   ./script.sh "custom_model/custom_LLM"
   ```
