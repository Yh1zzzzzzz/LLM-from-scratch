import torch
import json
import random
from pathlib import Path
from argparse import ArgumentParser
from unittest.mock import patch
from typing import List, Dict

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


    name = f"sft-ep{args.epochs}-lr{args.lr}-bs{args.batch_size}"
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
    #参数加载
    BASE_DIR = Path(__file__).resolve().parent.parent
    prompt = BASE_DIR / "cs336_alignment" / "prompts" / "r1_zero.prompt"
    math_dataset_path = BASE_DIR / "data" / "tulu-3-sft-personas-math" / "data" / "train-00000-of-00002.parquet"
    LLM_model = BASE_DIR / "data" / "models" / "Qwen2.5-Math-1.5B"
    training_data = BASE_DIR / "data" / "gsm8k"/ "train.jsonl"
    test_data = BASE_DIR / "data" / "gsm8k"/ "test.jsonl"

    #vllm参数设置
    

    row_training_data = load_jsonl(training_data)
    row_test_data = load_jsonl(test_data)
    sft_training_data = convert_math_to_sft_format(row_training_data)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_model, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(LLM_model, trust_remote_code=True)

    train_prompts = [item["prompt"] for item in sft_training_data]
    train_responses = [item["response"] for item in sft_training_data]
    tokenized_training = tokenize_prompt_and_output(train_prompts, train_responses, tokenizer)

    train_inputs_id = tokenized_training["input_ids"]
    train_labels = tokenized_training["labels"]
    train_response_mask = tokenized_training["response_mask"]


    print(f"Microbatch size: {microbatch_size}, Gradient accumulation steps: {gradient_accumulation_steps}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = (len(train_inputs_id) // args.batch_size) * args.epochs

    Lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)

    num_step = (len(train_inputs_id) // args.batch_size) * args.epochs
    
    for step in range(num_step):
        print(f"Step {step}")
        wandb_data = {}

        max_example = len(train_inputs_id)
        start_id = (step * microbatch_size) % max_example
        indexes = [(start_id + i) % max_example for i in range(microbatch_size)]

        index = torch.tensor(indexes)

        input_ids , labels, response_mask = sample(train_inputs_id, train_labels, train_response_mask, index, "cuda")

        response_log_probs = get_response_log_probs(model, input_ids, labels)

        loss, metadata = sft_microbatch_train_step(
            response_log_probs["log_probs"],
            response_mask,
            gradient_accumulation_steps,
            normalize_constant=1.0,
        )

        print(f"Loss: {loss.item()}")
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            Lr_scheduler.step()


                

        wandb_data["step"] = step
        wandb_data["lr"] = Lr_scheduler.get_last_lr()[0]
        wandb_data["loss"] = loss.item()

        wandb.log(wandb_data)

if __name__ == "__main__":
    main()

