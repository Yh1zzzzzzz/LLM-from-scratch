import torch
import json
import random
from pathlib import Path
from argparse import ArgumentParser
from unittest.mock import patch
from typing import List, Dict

# Transformers and Torch imports
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
import wandb
def load_jsonl(path: Path)->List[Dict[str,str]]:
    rows=[]
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows



def build_prompt(template_path: str, question: str) -> str:
    with open(template_path, 'r') as f:
        template = f.read()
    return template.replace("{question}", question)


def convert_math_to_sft_format(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    converted_rows = []
    for row in rows:
        question = row["question"]
        answer = row["answer"]
        BASE_DIR = Path(__file__).resolve().parent.parent
        prompt = BASE_DIR / "cs336_alignment" / "prompts" / "r1_zero.prompt" 
        # Format prompt using r1_zero template
        formatted_prompts = build_prompt(prompt, question) 
        
        converted_rows.append({
            "prompt": formatted_prompts,
            "response": answer
        })
    return converted_rows

def sample(train_input_ids, train_labels, train_response_mask, index, device):
    input_ids = torch.index_select(train_input_ids, 0, index).to(device)
    labels = torch.index_select(train_labels, 0, index).to(device)
    response_mask = torch.index_select(train_response_mask, 0, index).to(device)
    return (input_ids, labels, response_mask)

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer): 
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompt_tokens = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_tokens = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]
    concat_tokens = [p + o for p, o in zip(prompt_tokens, output_tokens)]
    max_len = max(len(t) for t in concat_tokens)
    dic = {}
    input_ids = []
    labels = []
    response_mask = []
    for pt, ot in zip(prompt_tokens, output_tokens):
        concat = pt + ot
        pad_len = max_len - len(concat)
        input_id = concat + [tokenizer.pad_token_id] * pad_len
        label = input_id[1:] + [tokenizer.pad_token_id]
        response_m = [False] * (len(pt) - 1) + [True] * len(ot) + [False] * (pad_len+1)
        input_ids.append(torch.tensor(input_id[:-1], dtype=torch.long))
        labels.append(torch.tensor(label[:-1], dtype=torch.long))
        response_mask.append(torch.tensor(response_m[:-1], dtype=torch.long))
    dic['input_ids'] = torch.stack(input_ids)
    dic['labels'] = torch.stack(labels)
    dic['response_mask'] = torch.stack(response_mask)
    return dic

def get_response_log_probs(model : PreTrainedModel,
                           input_ids : torch.Tensor,
                           labels : torch.Tensor,
                           return_token_entropys : bool = False
                           ) -> dict[str, torch.Tensor]:
            
    model = model.to("cuda")
    model.train()
    input_ids = input_ids.to("cuda")
    labels = labels.to("cuda")
    dic = {}
    outputs = model(input_ids = input_ids)
    logits = outputs.logits # (batch_size, seq_len, vocab_size)
    if return_token_entropys:
        dic["token_entropys"] = compute_entropy(logits) # (batch_size, seq_len)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    dic["log_probs"] = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # (batch_size, seq_len)
    return dic
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)

def masked_normalize(
tensor: torch.Tensor,
mask: torch.Tensor,
normalize_constant: float,
dim: int | None = None,
) -> torch.Tensor:
    masked_tensor = tensor * mask
    if dim is None:
        sum = masked_tensor.sum()
    else :
        sum = masked_tensor.sum(dim=dim, keepdim=True)
    return sum / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    per_token_loss = -policy_log_probs
    masked_loss = per_token_loss * response_mask
    batch_size = policy_log_probs.shape[0]
    deno = batch_size * normalize_constant * gradient_accumulation_steps
    loss = masked_loss.sum() / deno
    loss.backward()
    metadata = {
            "unmasked_loss": per_token_loss.mean().item(),
            "masked_loss": (masked_loss.sum() / response_mask.sum()).item(),
        }
    return loss, metadata