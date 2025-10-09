import torch
from transformers import PreTrainedModel
BASE_DIR = Path(__file__).resolve().parent.parent
LLM_model = BASE_DIR / "data" / "models" / "Qwen2.5-Math-1.5B"
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute the per-token entropy of a batch of logits.

    Args:
        logits: A float tensor of shape (batch_size, seq_len, vocab_size)
            representing the model logits.
            ogits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
containing unnormalized logits.
Returns:
torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
prediction.
Note: you should use a numerically stable method (e.g., using logsumexp) to avoid overflow.
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)
def get_response_log_probs(moodel : PreTrainedModel,
                           imput_ids : torch.Tensor,
                           labels : torch.Tensor,
                           return_token_entropys : bool = False
                           ) -> dict[str, torch.Tensor]:
    model = model.to("cuda")
    model.eval()
    imput_ids = imput_ids.to("cuda")
    labels = labels.to("cuda")
    dic = {}
    with torch.no_grad():
        outputs = model(input_ids = imput_ids， labels = labels)
        logits = outputs.logits # (batch_size, seq_len, vocab_size)
        if return_token_entropys:
            dic["token_entropys"] = compute_entropy(logits) # (batch_size, seq_len)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        dic["log_probs"] = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # (batch_size, seq_len)
    return dic

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
# Execute a forward-and-backward pass on a microbatch.
# Args:
# policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
# SFT policy being trained.
# response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
# prompt/padding.
# gradient_accumulation_steps Number of microbatches per optimizer step.
# normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.
# Returns:
# tuple[torch.Tensor, dict[str, torch.Tensor]].
# 12
# loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
# this so we can log it.
# metadata Dict with metadata from the underlying loss call, and any other statistics you
# might want to log.
# Implementation tips:
# • You should call loss.backward() in this function. Make sure to adjust for gradient
# accumulation.
    per_token_loss = -policy_log_probs
    masked_loss = per_token_loss * response_mask
    batch_size = policy_log_probs.shape[0]
    deno = batch_size * normalize_constant * gradient_accumulation_steps
    loss = masked_loss.sum() / deno
    loss.backward()
    with torch.no_grad():
        metadata = {
            "unmasked_loss": per_token_loss.mean().item(),
            "masked_loss": (masked_loss.sum() / response_mask.sum()).item(),
        }
    return loss, metadata

#----------------------------------------------------------------
# training
from vllm.model_executor import set_random_seed as vllm_set_random_seed
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
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def SFT():
    


