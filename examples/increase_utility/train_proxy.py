# Take a bunch of unlabeled examples, label with a gold model, and train a proxy model
import torch
import pandas as pd
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
                          AutoConfig, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from utils import get_scores
import pdb
# import GPUtil

"""
1. Load gold model and tokenizer
2. Load pretrained base for proxy model and tokenizer
3. Load unlabeled data
4. Score unlabeled data
5. Create training dataset 
6. Train proxy model 
7. Evaluate proxy model [Not implemented yet]
"""

def load_model(model_name: str, device: torch.device, checkpoint: str = None) -> AutoModelForSequenceClassification:
    # Load model
    config = AutoConfig.from_pretrained(model_name, num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    model.to(device)

    # Load pretrained checkpoint if available
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)

    return model


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_scenarios(load_data_path: str):
    df = pd.read_csv(load_data_path, header=None)
    scenarios = list(df.values.flatten())
    return scenarios


def tokenize_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer, device: torch.device) -> DatasetDict:
    return tokenizer(dataset["text"], padding="max_length", truncation=True, return_tensors="pt").to(device)


def create_dataset(scenarios: List[str], scores: torch.Tensor, tokenizer: AutoTokenizer, device: torch.device) -> Dataset:
    df = pd.DataFrame({"text": scenarios, "label": scores.detach().cpu().numpy()})
    dataset = DatasetDict({"train": Dataset.from_pandas(df, preserve_index=False)})
    tokenized_dataset = dataset.map(tokenize_dataset, batched=True, fn_kwargs={"tokenizer": tokenizer, "device": device})
    # TODO: Save another split of this data for evaluation. 
    # TODO: Should we save this to disk or just train the proxy model? 
    return tokenized_dataset


def compute_metrics(eval_pred):
    # TODO: Use this for evaluation during training
    preds, labels = eval_pred
    loss_fn = torch.nn.MSELoss()
    return loss_fn(preds, labels)


if __name__=="__main__":
    # TODO: Parse CLI arguments

    # Load gold model 
    gold_model_name = "roberta-large"
    checkpoint = "models/reward/util_roberta-large.pt"
    gold_device = torch.device("cuda:0")
    gold_model = load_model(gold_model_name, gold_device, checkpoint)
    gold_tokenizer = load_tokenizer(gold_model_name)

    # Load pretrained base for proxy model 
    proxy_model_name = "roberta-large"
    # proxy_model_name = "smallbenchnlp/roberta-small"
    proxy_device = torch.device("cuda:2")
    proxy_model = load_model(proxy_model_name, proxy_device)
    proxy_tokenizer = load_tokenizer(proxy_model_name)

    # Load unlabeled data
    load_data_path = "data/unlabeled_scenarios/0.csv"
    scenarios = load_scenarios(load_data_path)

    # Score scenarios with gold model
    scores = get_scores(scenarios, gold_device, gold_tokenizer, gold_model)

    # Create Hugging Face dataset
    train_dataset = create_dataset(scenarios, scores, proxy_tokenizer, proxy_device)['train']
    # TODO: Which device is this dataset on? Are we safe? 
    
    print()
    print()
    print("beginning training")
    print()
    print()
        
    # Train proxy model
    training_args = TrainingArguments(
        output_dir="proxy_trainer", 
        evaluation_strategy="no", 
        remove_unused_columns=False,
        per_device_train_batch_size=1
    )
    trainer = Trainer(
        model=proxy_model,
        args=training_args,
        train_dataset=train_dataset
    )
    trainer.train()

    # TODO: Evaluate proxy model
    print("done")
