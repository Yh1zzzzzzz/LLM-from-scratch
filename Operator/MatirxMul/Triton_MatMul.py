import triton
import triton.language as tl
@triton.jit
def matmul(A, B, C, M, N, K, stride_am, stride_ak, 
            stride_bk, stride_bn, stride_cm, stride_cn,
            **META):
    # extract metaparameters
    BLOCK_M, GROUP_M = META['BLOCK_M'], META['GROUP_M']
    BLOCK_N = META['BLOCK_N']
    BLOCK_K = META['BLOCK_K']
    # programs are grouped together to improve L2 hit rate
    _pid_m = tl.program_id(0)
    _pid_n = tl.program_id(1)
    pid_m = _pid_m // GROUP_M
    pid_n = (_pid_n * GROUP_M) + (_pid_m % GROUP_M)
    # rm (resp. rn) denotes a range of indices
    # for rows (resp. col) of C
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # rk denotes a range of indices for columns 
    # (resp. rows) of A (resp. B)
    rk = tl.arange(0, BLOCK_K)
    # the memory addresses of elements in the first block of
    # A and B can be computed using numpy-style broadcasting
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk [:, None] * stride_bk  + rn[None, :] * stride_bn)
    # initialize and iteratively update accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K, 0, -BLOCK_K):
        a = tl.load(A)
        b = tl.load(B)
        # block level matrix multiplication
        acc += tl.dot(a, b)
        # increment pointers so that the next blocks of A and B
        # are loaded during the next iteration
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    # acc = tl.where(acc >= 0, acc, alpha * acc)
    # write back result
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C, acc, mask=mask)