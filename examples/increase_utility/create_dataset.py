from typing import List, Tuple
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_from_disk

def get_prompt_label(index: int, sentence: str) -> Tuple[str, str]:
    prompt = sentence[:index]
    label = sentence[index:].strip(" ")
    return prompt, label

def get_rand_start_end(length: int, start_val: float=0.3, end_val: float=0.7) -> int:
    rand_start = int(start_val * length)
    rand_end = int(end_val * length)
    rand_int = random.randint(rand_start, rand_end)
    return rand_int

def get_all_prompts_labels(positive_examples: List[str]) -> Tuple[List[str], List[str]]:
    all_prompts = []
    all_labels = []
    for example in positive_examples:
        number_sentences = example.count(".")
        if number_sentences <= 1:
            words = example.split(" ")
            length = len(words)
            rand_int = get_rand_start_end(length)
            index = example.index(words[rand_int])
            prompt, label = get_prompt_label(index, example)
        else:
            period_index = example.index(".")
            prompt, label = get_prompt_label(period_index+1, example)
        all_prompts.append(prompt)
        all_labels.append(label)
    return all_prompts, all_labels

def create_df(prompts: List[str], labels: List[str]) -> pd.DataFrame:
    return pd.DataFrame({"prompt":prompts, "label":labels})

def transform_data(csv_path: str) -> pd.DataFrame:
    original_df = pd.read_csv(csv_path, header=None)
    positive_utilities = original_df.iloc[:,[0,1]].values.reshape((-1))
    prompts, labels = get_all_prompts_labels(positive_utilities)
    transformed_df = create_df(prompts, labels)
    return transformed_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="ethics/utilitarianism/util_train.csv", help="path to load utility train file")
    parser.add_argument("--test_file", type=str, default="ethics/utilitarianism/util_test.csv", help="path to load utility test file")
    parser.add_argument("--dataset_path", type=str, default="utility_dataset.hf", help="path to save HuggingFace dataset of continuation pairs")
    args = parser.parse_args()

    train_val_df = transform_data(args.train_file)
    train_df, val_df = train_test_split(train_val_df.reset_index(drop=True), test_size = 0.1)
    test_df = transform_data(args.test_file)
    full_dataset = DatasetDict({"train": Dataset.from_pandas(train_df, preserve_index=False), 
                                "val": Dataset.from_pandas(val_df, preserve_index=False),
                                "test": Dataset.from_pandas(test_df)})
    full_dataset.save_to_disk(args.dataset_path)
    print("dataset saved")
