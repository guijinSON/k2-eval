import argparse
from datasets import load_dataset
import pandas as pd
from outlines import models
import outlines
import torch
from template import gen_template, kno_template, constraints
from vllm import LLM, SamplingParams

def load_data(subset):
    return pd.DataFrame(load_dataset("HAERAE-HUB/K2-Eval", subset)['test'])

def initialize_llm(model_path):
    llm = LLM(model=model_path, #max_model_len=4096, 
              tensor_parallel_size=torch.cuda.device_count())#,token="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")
    model = models.VLLM(llm)
    generator = outlines.generate.choice(model, ["A","B","C","D"])
    return generator, llm

def generate_answers(llm, prompts, sampling_params):
    answers = llm.generate(prompts, sampling_params)
    return [answer.outputs[0].text.strip() for answer in answers]

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text based on specified models and constraints")
    parser.add_argument("--model_path", type=str, required=True, help="The path or identifier for the model to use")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load data
    gen_data = load_data("generation")
    kno_data = load_data("knowledge")
    
    # Initialize model with command line argument
    generator, llm = initialize_llm(args.model_path)
    
    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7, 
        top_p=0.95,
        min_tokens=20,
        max_tokens=1600,
        stop=['###']
    )
    
    # Generate answers for generation data
    gen_prompts = [gen_template.format(instruction) for instruction in gen_data.instruction]
    gen_answers = generate_answers(llm, gen_prompts, sampling_params)
    gen_data['generation'] = gen_answers
    
    # Merge knowledge data with generation results
    merged_data = gen_data.merge(kno_data, on=['instruction'])
    
    # Generate answers for knowledge data
    kno_prompts = [kno_template.format(row.instruction, row.generation, row.question, row.a, row.b, row.c, row.d) for _, row in merged_data.iterrows()]
    kno_answers = generator(kno_prompts)
    merged_data['predict'] = kno_answers
    
    # Apply constraints
    constraint_data = pd.DataFrame([(gen, constraint, template.format(gen)) for gen in gen_answers for constraint,template in constraints.items()], columns=['generation','constraint','query'])
    constraint_results = generate_answers(llm, constraint_data['query'].values, sampling_params)
    constraint_data['regeneration'] = constraint_results
    
    # Output to CSV
    model_path = args.model_path#.split('/')[1]
    gen_data.to_csv(f'{model_path}_gen.csv')
    merged_data.to_csv(f'{model_path}_kno.csv')
    constraint_data.to_csv(f'{model_path}_con.csv')

if __name__ == "__main__":
    main()
