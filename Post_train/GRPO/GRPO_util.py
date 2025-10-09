
import torch
import json
import random
from pathlib import Path
from argparse import ArgumentParser
from unittest.mock import patch
from typing import List, Dict
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
# Transformers and Torch imports
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
import wandb
import json 
from vllm import LLM, SamplingParams
from typing import Callable, Literal

def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    metadata = {} #存储std min/max等信息
    return_rewards = torch.zeros(len(rollout_responses), 1)
    return_unnormalized = torch.zeros(len(rollout_responses), 1)
    batch_size = len(rollout_responses) // group_size #每个批次的提示数
    for i in range(int(len(rollout_responses))):
        reward = reward_fn(rollout_responses[i], repeated_ground_truths[i])["reward"]
        return_unnormalized[i] = reward
        return_rewards[i] = reward
    return_rewards_grouped = return_rewards.view(batch_size, group_size)
    mean_rewards = return_rewards_grouped.mean(dim=1, keepdim=True)
    if normalize_by_std:
        std_rewards = return_rewards_grouped.std(dim=1, keepdim=True)
        metadata["std_rewards"] = std_rewards.mean().item()
        return_rewards_grouped = (return_rewards_grouped - mean_rewards) / (std_rewards + advantage_eps)
        metadata["min_reward"] = return_rewards_grouped.min().item()
        metadata["max_reward"] = return_rewards_grouped.max().item()
    else:
        return_rewards_grouped = return_rewards_grouped - mean_rewards
        metadata["min_reward"] = return_rewards_grouped.min().item()
        metadata["max_reward"] = return_rewards_grouped.max().item()
    return return_rewards_grouped.view(-1), return_unnormalized.view(-1), metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor, #(batch_size, 1)
    policy_log_probs: torch.Tensor,           #(batch_size, seq_len)
    ) -> torch.Tensor:
    loss = - (raw_rewards_or_advantages * log_probs_sum)
    return loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metadata = {}
    ratio = torch.exp(policy_log_probs - old_log_probs) #(batch_size, seq_len)
    #clip the ratio
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    min_advantage = torch.min(ratio * advantages, clipped_ratio * advantages)
    loss = - min_advantage

    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,#(batch_size, seq_len)
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None, #batch_size x 1
    advantages: torch.Tensor | None = None, #batch_size x 1
    old_log_probs: torch.Tensor | None = None,#batch_size x seq_len
    cliprange: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metadata = {}
    if loss_type == "no_baseline":
        assert raw_rewards is not None  
        loss = - (raw_rewards * policy_log_probs)
        return loss, metadata
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None  
        loss = - (advantages * policy_log_probs)
        return loss, metadata
    elif loss_type == "grpo_clip":
        assert advantages is not None and old_log_probs is not None and cliprange is not None
        ratio = torch.exp(policy_log_probs - old_log_probs) #(batch_size, seq_len)
        #clip the ratio
        clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

        min_advantage = torch.min(ratio * advantages, clipped_ratio * advantages)
        loss = - min_advantage

        return loss, metadata
def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    ) -> torch.Tensor:
    if dim is None:
        masked_tensor = tensor * mask
        return  masked_tensor.sum() / mask.sum()
    else:
        masked_tensor = tensor * mask
        return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,#(batch_size, sequence_length)
    response_mask: torch.Tensor,#(batch_size, sequence_length)
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
# Args:
# policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
# policy being trained.
# response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
# prompt/padding.
# gradient_accumulation_steps Number of microbatches per optimizer step.
# loss_type One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
# raw_rewards Needed when loss_type == "no_baseline"; shape (batch_size, 1).
# advantages Needed when loss_type != "no_baseline"; shape (batch_size, 1).
# old_log_probs Required for GRPO-Clip; shape (batch_size, sequence_length).
# cliprange Clip parameter ϵ for GRPO-Clip.
# Returns:
# tuple[torch.Tensor, dict[str, torch.Tensor]].
# loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
# this so we can log it.
# metadata Dict with metadata from the underlying loss call, and any other statistics you
# might want to log.

    if loss_type == "no_baseline":
        assert raw_rewards is not None
        # per-token loss, then masked mean over seq
        per_token_loss = -(raw_rewards * policy_log_probs)  # (B, T)
        loss_per_example = masked_mean(per_token_loss, response_mask, dim=1)  # (B,)
        loss = loss_per_example.mean() / gradient_accumulation_steps
        loss.backward()
        return loss, {}

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        per_token_loss = -(advantages * policy_log_probs)  # (B, T)
        loss_per_example = masked_mean(per_token_loss, response_mask, dim=1)  # (B,)
        loss = loss_per_example.mean() / gradient_accumulation_steps
        loss.backward()
        return loss, {}

    else:
        assert advantages is not None and old_log_probs is not None and cliprange is not None
        # per-token GRPO-Clip
        ratio = torch.exp(policy_log_probs - old_log_probs)  # (B, T)
        clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
        per_token_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)  # (B, T)
        loss_per_example = masked_mean(per_token_loss, response_mask, dim=1)  # (B,)
        loss = loss_per_example.mean() / gradient_accumulation_steps
        loss.backward()
        return loss, {}
    

@torch.no_grad() # 明确告诉PyTorch在函数内不需要计算梯度
def generate_answers(model, tokenizer, prompts, device, max_new_tokens=256):
    """
    使用当前模型为给定的prompts生成答案。
    
    Args:
        model: 当前训练的模型
        tokenizer: 分词器
        prompts (list of str): 用于生成答案的prompt列表
        device: 'cuda' 或 'cpu'
        max_new_tokens (int): 生成答案的最大长度
        
    Returns:
        list of str: 生成的答案列表
    """
    # 切换到评估模式，这会禁用dropout等训练特有的层
    model.eval()
    
    outputs = []
    for prompt in prompts:
        # 准备输入
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 生成答案
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id, # 防止生成padding
            do_sample=False # 使用确定性生成（greedy search）以保证结果可复现
        )
        
        # 解码生成的token，并去除原始的prompt部分
        full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # 一个简单的方法去除prompt，对于复杂情况可能需要更鲁棒的逻辑
        generated_answer = full_text[len(prompt):].strip()
        outputs.append(generated_answer)
        
    # 切换回训练模式，非常重要！
    model.train()
    return outputs