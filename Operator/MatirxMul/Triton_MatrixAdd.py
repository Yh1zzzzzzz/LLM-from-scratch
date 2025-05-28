import torch
from typing import List
import triton
import triton.language as tl
@triton.jit
def MatAdd(
    A,
    B, 
    C,
    stride_am,
    stride_an,
    stride_bm,
    stride_bn,
    stride_cm,
    stride_cn,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 2D block索引
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算当前block的行列范围
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 边界检查
    mask_m = rows < M
    mask_n = cols < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # 计算内存偏移
    A_ptrs = A + rows[:, None] * stride_am + cols[None, :] * stride_an
    B_ptrs = B + rows[:, None] * stride_bm + cols[None, :] * stride_bn
    C_ptrs = C + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    
    # 加载、计算、存储
    _A = tl.load(A_ptrs, mask=mask, other=0.)
    _B = tl.load(B_ptrs, mask=mask, other=0.)
    _C = _A + _B
    tl.store(C_ptrs, _C, mask=mask)

def matrix_add(A, B):
    C = torch.empty_like(A)
    BLOCK_M, BLOCK_N = 32, 32
    
    grid = (triton.cdiv(A.shape[0], BLOCK_M), 
            triton.cdiv(A.shape[1], BLOCK_N))
    
    MatAdd[grid](
        A, B, C,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), 
        C.stride(0), C.stride(1),
        A.shape[0], A.shape[1],
        BLOCK_M, BLOCK_N
    )
    return C