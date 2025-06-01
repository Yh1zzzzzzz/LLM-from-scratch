import torch
import torch.nn.functional as F
import math
import time

# 确保 CUDA 可用
if not torch.cuda.is_available():
    print("CUDA is not available. Performance comparison will be on CPU and may not be meaningful for FlashAttention.")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda")

def naive_sdpa_implementation(query, key, value, is_causal=False, scale=None):
    batch_size, num_heads, q_seq_len, head_dim = query.shape
    _, _, kv_seq_len, _ = key.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    if is_causal:
        mask = torch.triu(torch.ones(q_seq_len, kv_seq_len, device=query.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attention_weights, value)
    return output

compiled_sdpa = torch.compile(naive_sdpa_implementation, mode="max-autotune", fullgraph=True)
