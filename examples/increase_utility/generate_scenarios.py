# Generates a dataset of ethical scenarios
# Rate these with the gold model and then train the reward model
import torch
import argparse
from typing import List, Tuple 
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import pdb
import os   
import pandas as pd

def few_shot_generation(n: int, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                        device: torch.device, debug: bool) -> List[Tuple[str]]:
    
    examples = []
    
    first_prompt = """Factual descriptions of positive and negative scenarios. 
    Scenario #1
    More pleasant: I had to give a speech for class. My friends applauded for me.
    Less pleasant: I had to give a speech for class. I cried in the middle of it.
    Scenario #2
    More pleasant: I cooked breakfast. It was good.
    Less pleasant: I cooked breakfast. I threw it up after.
    Scenario #3
    More pleasant: I go to buy a new saxophone from the instrument shop.
    Less pleasant: I go to buy a new saxophone from the instrument shop. There are no saxophones left at the shop.
    Scenario #4
    More pleasant: I made my own coffee this morning. It was ok.
    Less pleasant: I spilled hot coffee on my hand this morning.
    Scenario #5
    More pleasant:"""
    
    for _ in range(n):
        # Generate the first completion
        more_pleasant = generate(first_prompt, model, device, stop_token="\n")
        
        # Generate the second completion using a new prompt
        second_prompt = first_prompt + more_pleasant + "\nLess pleasant:"
        less_pleasant = generate(second_prompt, model, device, stop_token="\n")
        
        examples.append(tuple([more_pleasant, less_pleasant]))
        
        if debug:
            print(f"More pleasant: {more_pleasant}")
            print(f"Less pleasant: {less_pleasant}")
    
    return examples
    
def generate(prompt: str, model, device: torch.device, stop_token: str) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        input_ids, 
        do_sample=True, 
        max_length=len(input_ids[0])+30, 
        top_p=0.95, 
        top_k=0,
        pad_token_id=tokenizer.eos_token_id
    )
    output = tokenizer.decode(output[0], skip_special_tokens=True)
                
    # Remove the original prompt if present
    if len(output) > len(prompt):
        output = output[len(prompt):]
    
    # Get rid of anything past the first occurrence of stop_token
    output = output[:output.find(stop_token)]

    return output


def get_save_path():
    save_dir = 'data/unlabeled_scenarios'
    file_name = str(len([name for name in os.listdir(save_dir)
                    if os.path.isfile(os.path.join(save_dir, name))]))
    full_path = save_dir + '/' + file_name + '.csv'
    return full_path


def save_to_csv(examples):
    df = pd.DataFrame(examples)
    save_path = get_save_path()
    df.to_csv(save_path, index=False, header=False)
    print(f"Saved as {save_path}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2-large", help="model to generate scenarios")
    parser.add_argument('--debug', action='store_true', help='enable debugging (default: False)')
    parser.add_argument("--n", type=int, default=5, help="number of example pairs to generate")
    args = parser.parse_args()
    
    device = torch.device("cuda:{}".format(0)) 
    tokenizer = AutoTokenizer.from_pretrained(args.model, device=device)
    tokenizer.pad_token = tokenizer.eos_token
    if "t5" in args.model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    print("loaded generator model")
    
    examples = few_shot_generation(args.n, model, tokenizer, device, args.debug)
    
    save_to_csv(examples)