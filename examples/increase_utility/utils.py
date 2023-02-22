import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

def get_scores(
    samples: List[str], 
    device: torch.device, 
    tokenizer: AutoTokenizer, 
    model: AutoModelForSequenceClassification, 
    max_length: int = 512, 
    batch_size: int = 2
) -> torch.tensor:
    """
    Score a list of strings using the provided reward model. Returns a tensor of scores. 
    """
    scores_list = []
    for i in range(0, len(samples), batch_size):
        sub_samples = samples[i: i + batch_size]
        encodings_dict = tokenizer(
            sub_samples,
            truncation=True,
            max_length=max_length, # Different, but I think I'm right. 
            padding=True,
            return_tensors="pt",
        )
        input_ids = encodings_dict["input_ids"].to(device)
        attn_masks = encodings_dict["attention_mask"].to(device)
        input_ids = input_ids.repeat(2, 1)
        attn_masks = attn_masks.repeat(2, 1)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attn_masks)
            sub_scores = output.logits.reshape((-1))
        scores_list.append(sub_scores)
    scores = torch.cat(scores_list, dim=0)
    return scores