import os
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from argparse import ArgumentParser, Namespace

# Argument parser
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, default="42dot/42dot_LLM-SFT-1.3B", help="Model identifier")
    parser.add_argument("--dataset", type=str, default="HAERAE-HUB/KOREAN-WEBTEXT", help="Dataset identifier")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for evaluation results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--n_gpu", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use")
    parser.add_argument("--csv_filename", type=str, default="perplexity.csv", help="Filename to save token counts")
    return parser.parse_args()

# Example dataset class
class EvalDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        return input_ids

def collate(examples: List[torch.Tensor], tokenizer):
    if tokenizer._pad_token is None:
        return pad_sequence(examples, batch_first=True)
    return pad_sequence(
        examples, batch_first=True, padding_value=tokenizer.pad_token_id
    )

def evaluate(model, tokenizer, eval_dataset, batch_size, device, n_gpu):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=eval_sampler,
        collate_fn=lambda x: collate(x, tokenizer),
    )

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Running Evaluation"):
        inputs, labels = (batch, batch)
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            if n_gpu > 1:
                lm_loss = lm_loss.mean()
            eval_loss += lm_loss.item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    results["eval_loss"] = eval_loss
    results["perplexity"] = perplexity

    return results

def save_to_csv(model_name, results, filename="token_count.csv"):
    new_data = pd.DataFrame({
        "model_name": [model_name],
        "eval_loss": [results["eval_loss"]],
        "perplexity": [results["perplexity"]]
    })

    if os.path.exists(filename):
        existing_data = pd.read_csv(filename)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        updated_data.to_csv(filename, index=False)
    else:
        new_data.to_csv(filename, index=False)

def main():
    args = parse_args()

    # Initialize model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_id).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Load dataset
    input_text = load_dataset(args.dataset, split=args.split)['text'][:12800]

    # Create evaluation dataset
    eval_dataset = EvalDataset(tokenizer, input_text)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate
    results = evaluate(model, tokenizer, eval_dataset, args.batch_size, device, args.n_gpu)

    # Print and save results
    print(results)

    # Save token count to CSV
    save_to_csv(args.model_id, results, args.csv_filename)

if __name__ == "__main__":
    main()
