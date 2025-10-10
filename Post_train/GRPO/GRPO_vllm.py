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

BASE_DIR = Path(__file__).resolve().parent.parent
prompt = BASE_DIR / "cs336_alignment" / "prompts" / "r1_zero.prompt"
math_dataset_path = BASE_DIR / "data" / "tulu-3-sft-personas-math" / "data" / "train-00000-of-00002.parquet"
LLM_model = BASE_DIR / "data" / "models" / "Qwen2.5-Math-1.5B"

def build_prompt(template_path: str, question: str) -> str:
    with open(template_path, 'r') as f:
        template = f.read()
    return template.replace("{question}", question)

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    answers: List[str]
) -> None:
    """
    评估语言模型在一系列提示（prompts）上的表现，
    计算评估指标，并将结果序列化到磁盘。
    """
    sample_param = eval_sampling_params
    prompt_outputs = vllm_model.generate(prompts, sampling_params=sample_param)
    results = []
    for i, output in enumerate(prompt_outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        reward = reward_fn(generated_text, answers[i])
        results.append({
            "prompt": prompt,
            "output": generated_text,
            "reward": reward,
        })
    with open("evaluation_results.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n" + '\n')
            f.write('\n')



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
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float=0.85):
    """"Start the inference process, here we use vLLM to hold a model on
        a GPU separate from the policy.

    """
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
    return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
        model=model_id,
        device=device,
        dtype=torch.bfloat16,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
        )
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
        question = row["prompt"]
        answer = row["response"]
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



def main():
    SEED = 42
    torch.manual_seed(SEED)
    random.seed(SEED)
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int,default=16)
    parser.add_argument("--microbatch-size", type=int,  default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--eval-every-n-batches", type=int, default=10)
    parser.add_argument("--filter-out-incorrect-training-data", action="store_true")
    args = parser.parse_args()
    if args.batch_size % args.microbatch_size != 0:
        raise ValueError(
            f"batch_size ({args.batch_size}) must be divisible by microbatch_size ({args.microbatch_size})"
        )
    gradient_accumulation_steps = args.batch_size // args.microbatch_size
    microbatch_size = args.microbatch_size


    name = f"GRPO-Qwen2.5{args.epochs}-lr{args.lr}-bs{args.batch_size}"
    run = wandb.init(
        name=name,
        config={
            "epochs": args.epochs,
            "lr": args.lr,
            "gradient_accum_steps": gradient_accumulation_steps,
            "microbatch_size": microbatch_size,
        },
    )


    device = "cuda"
    BASE_DIR = Path(__file__).resolve().parent.parent
    prompt = BASE_DIR / "cs336_alignment" / "prompts" / "r1_zero.prompt"
    math_dataset_path = BASE_DIR / "data" / "tulu-3-sft-personas-math" / "data" / "train-00000-of-00002.parquet"
    LLM_model = BASE_DIR / "data" / "models" / "Qwen2.5-Math-1.5B"
    training_data = BASE_DIR / "data" / "gsm8k"/ "gsm8k_train.jsonl"
    test_data = BASE_DIR / "data" / "gsm8k"/ "gsm8k_test.jsonl"
    model_save_path = BASE_DIR / "GRPO_model" / "GRPO_models"

    # Hyperparameters
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 16
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4 # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1 # On-policy
    train_batch_size: int = 8 # On-policy
    gradient_accumulation_steps: int = 4 # microbatch size is 2, will fit on H100
    gpu_memory_utilization: float = 0.85
    loss_type: Literal[
    "no_baseline",
    "reinforce_with_baseline",
    "grpo_clip",
    ] = "reinforce_with_baseline"
    use_std_normalization: bool = True
    
    
    assert train_batch_size % gradient_accumulation_steps == 0, (
    "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
    "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
    "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size


    #训练数据集准备
    row_training_data = load_jsonl(training_data)
    row_test_data = load_jsonl(test_data)
    sft_training_data = convert_math_to_sft_format(row_training_data) #转换为Cot格式
    sft_test_data = convert_math_to_sft_format(row_test_data) #转换为Cot格式

    train_prompts = [item["prompt"] for item in sft_training_data]#训练所需的prompt
    train_responses = [item["response"] for item in sft_training_data]#训练所需的response
    


    #模型准备
    model = AutoModelForCausalLM.from_pretrained(
        LLM_model, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        trust_remote_code=True
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(LLM_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ref_policy = AutoModelForCausalLM.from_pretrained(
    #     LLM_model, 
    #     torch_dtype=torch.bfloat16, 
    #     attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    #     trust_remote_code=True
    # ).to(device)
    # ref_policy.eval() # 冻结参考模型

    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.0,
    betas=(0.9, 0.95),
    )
    
    Lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_grpo_steps)

    #==================================
    #Vllm for inference
    #==================================
    print("Initializing vLLM engine for fast rollouts...")
    vllm_engine = LLM(model=str(LLM_model), trust_remote_code=True, seed=SEED)
    sampling_params = SamplingParams(
        n=group_size,
        temperature=sampling_temperature,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
    )
    print("vLLM engine initialized.")


    

    #日志
    generations_output_file = BASE_DIR / f"{name}_generations.jsonl"
    # 在训练开始前，如果文件已存在，则删除它，以确保每次运行都是全新的记录
    if generations_output_file.exists():
        generations_output_file.unlink()
    print(f"每100步的生成结果将保存在: {generations_output_file}")
    print(f"Microbatch size: {microbatch_size}, Gradient accumulation steps: {gradient_accumulation_steps}")
    # num_training_steps = (len(train_inputs_id) // args.batch_size) * args.epochs


    num_step = n_grpo_steps
    global_step = 0
    for step in range(num_step):
        print(f"\n===== GRPO Step {step}/{n_grpo_steps} =====")
        wandb_data = {"grpo_step": step}


        print(f"Generating {rollout_batch_size} responses from {n_prompts_per_rollout_batch} prompts...")

        prompt_indices = random.sample(range(len(train_prompts)), n_prompts_per_rollout_batch)
        batch_prompts = [train_prompts[i] for i in prompt_indices]
        batch_ground_truths = [train_responses[i] for i in prompt_indices]

        vllm_outputs = vllm_engine.generate(batch_prompts, sampling_params)

        rollout_responses = []
        for output in vllm_outputs:
            rollout_responses.extend([comp.text for comp in output.outputs])
        
        repeated_ground_truths = [gt for gt in batch_ground_truths for _ in range(group_size)]

        advantages, unnormalized_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )
        #=================
        #注意这里 advantages 和 unnormalized_rewards的shape是(batch_size, 1)
        advantages = advantages.to(device)
        wandb_data.update(reward_metadata)
        wandb_data["unnormalized_reward_mean"] = unnormalized_rewards.mean().item()

        # 将 prompts 和生成的 responses 组合并分词
        repeated_prompts = [p for p in batch_prompts for _ in range(group_size)]
        tokenized_rollouts = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)

        if loss_type == "grpo_clip":
            with torch.no_grad():
                old_log_probs_dict = get_response_log_probs(
                    ref_policy, 
                    tokenized_rollouts["input_ids"], 
                    tokenized_rollouts["labels"]
                )
                old_log_probs = old_log_probs_dict["log_probs"].to(device)
        else:
            old_log_probs = None # 其他损失类型不需要

        for epoch in range(epochs_per_rollout_batch):
            # 将 rollout 数据分批
            indices = list(range(rollout_batch_size))
            random.shuffle(indices)
            
            for i in range(0, rollout_batch_size, train_batch_size):
                optimizer.zero_grad()
                for j in range(0, train_batch_size, micro_train_batch_size):
                    global_step += 1
                    micro_batch_indices = indices[i + j : i + j + micro_train_batch_size]
                    if not micro_batch_indices: continue

                    # 准备 micro batch 的数据
                    input_ids = tokenized_rollouts["input_ids"][micro_batch_indices].to(device)
                    labels = tokenized_rollouts["labels"][micro_batch_indices].to(device)
                    response_mask = tokenized_rollouts["response_mask"][micro_batch_indices].to(device)
                    micro_batch_advantages = advantages[micro_batch_indices].unsqueeze(-1)
                    micro_batch_rewards = unnormalized_rewards[micro_batch_indices].unsqueeze(-1).to(device)
                    micro_batch_old_log_probs = old_log_probs[micro_batch_indices] if old_log_probs is not None else None

                    # 核心步骤：用当前策略计算 log_probs
                    policy_log_probs_dict = get_response_log_probs(model, input_ids, labels)
                    policy_log_probs = policy_log_probs_dict["log_probs"]

                    # 计算损失并反向传播
                    loss, metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=response_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        loss_type=loss_type,
                        raw_rewards=micro_batch_rewards,
                        advantages=micro_batch_advantages,
                        old_log_probs=micro_batch_old_log_probs,
                        cliprange=0.15, # for grpo_clip
                    )
                    
                    print(f"  Micro Step {global_step}, Loss: {loss.item():.4f}")
                    wandb_data["loss"] = loss.item()

                # 完成一个 full batch 的梯度累积后，更新模型
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # 更新学习率
        Lr_scheduler.step()
        wandb_data["lr"] = Lr_scheduler.get_last_lr()[0]
        
        # 更新参考策略 (for GRPO-Clip) 和 vLLM 引擎 (可选但推荐)
        if loss_type == "grpo_clip":
            print("Updating reference policy...")
            ref_policy.load_state_dict(model.state_dict())
        
        #实现 vLLM 引擎的权重同步逻辑
        if step % 10 == 0: # 例如每 10 步同步一次
            
            model.save_pretrained(model_save_path)
            del vllm_engine
            torch.cuda.empty_cache()
            vllm_engine = LLM(model=str(model_save_path), trust_remote_code=True, seed=SEED)

        #### 定期评估  ####
        if step > 0 and step % 10 == 0: # 每 10 个 GRPO step 评估一次
            print(f"\n--- Step {step}: Generating answers for evaluation ---")
            
            k = 2 
            eval_indices = random.sample(range(len(sft_test_data)), k)
            eval_prompts = [sft_test_data[i]["prompt"] for i in eval_indices]
            gold_answers = [sft_test_data[i].get("response") for i in eval_indices]

            generated_answers = generate_answers(model, tokenizer, eval_prompts, device)
            
            with open(generations_output_file, "a", encoding="utf-8") as f:
                for prompt, answer, gold in zip(eval_prompts, generated_answers, gold_answers):
                    record = {"step": step, "prompt": prompt, "generated_answer": answer, "gold_answer": gold}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            print(f"--- Saved {len(eval_prompts)} generated answers to {generations_output_file} ---")
        
        wandb.log(wandb_data)


if __name__ == "__main__":
    main()
    
