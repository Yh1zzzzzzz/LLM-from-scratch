# 主播已被Flash Attention和Triton折磨到破防不想继续debug了😃，

## torch.compile实现与 官方实现的区别：
Using device: cuda
PyTorch version: 2.5.1+cu124
CUDA version: 12.4
SDPA flash attention available: True
SDPA mem efficient attention available: True
SDPA math fallback available: True
Compiling custom SDPA (first call may be slow)...
Compilation done or already cached.
Benchmarking Compiled SDPA (Custom) (is_causal=False)...
Compiled SDPA (Custom) (is_causal=False): 1.5228 ms per run
Benchmarking Compiled SDPA (Custom) (is_causal=True)...
Compiled SDPA (Custom) (is_causal=True): 2.6953 ms per run
------------------------------
Benchmarking Official F.sdp_attention (is_causal=False)...
Official F.sdp_attention (is_causal=False): 1.2989 ms per run
Benchmarking Official F.sdp_attention (is_causal=True)...
Official F.sdp_attention (is_causal=True): 0.7550 ms per run
------------------------------
Verifying correctness (causal)...
Correctness check PASSED for causal attention.

Verifying correctness (non-causal)...
Correctness check PASSED for non-causal attention.

--- Benchmarking F.sdp_attention with different backends (if Pytorch 2.1+) ---

