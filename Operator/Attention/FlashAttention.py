import torch
import math
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _flash_attention_forward_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    L_ptr, M_ptr,  # L: 归一化因子, M: 最大值
    q_batch_stride, q_head_stride, q_seq_stride, q_dim_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_dim_stride,
    v_batch_stride, v_head_stride, v_seq_stride, v_dim_stride,
    o_batch_stride, o_head_stride, o_seq_stride, o_dim_stride,
    l_batch_stride, l_head_stride, l_seq_stride,
    m_batch_stride, m_head_stride, m_seq_stride,
    N_CTX, N_HEAD, N_DIM, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, CAUSAL: tl.constexpr,
):
    # 获取当前块的索引
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # 计算batch和head索引
    off_b = off_hz // N_HEAD
    off_h = off_hz % N_HEAD
    
    # 初始化指针偏移
    q_offset = off_b * q_batch_stride + off_h * q_head_stride
    k_offset = off_b * k_batch_stride + off_h * k_head_stride
    v_offset = off_b * v_batch_stride + off_h * v_head_stride
    o_offset = off_b * o_batch_stride + off_h * o_head_stride
    l_offset = off_b * l_batch_stride + off_h * l_head_stride
    m_offset = off_b * m_batch_stride + off_h * m_head_stride
    
    # 计算序列索引
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # 初始化累加器
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    
    # 加载查询Q
    q_ptrs = Q_ptr + q_offset + (offs_m[:, None] * q_seq_stride + offs_d[None, :] * q_dim_stride)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX)
    
    # 缩放因子
    scale = 1.0 / math.sqrt(BLOCK_DMODEL)
    q = q * scale
    
    # 遍历所有K、V块
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # 计算当前块的索引
        offs_n_curr = start_n + offs_n
        
        # 因果掩码：只处理当前或之前的token
        if CAUSAL:
            # 检查是否需要处理这个块
            if start_n > start_m * BLOCK_M + BLOCK_M - 1:
                break
        
        # 加载键K和值V
        k_ptrs = K_ptr + k_offset + (offs_n_curr[:, None] * k_seq_stride + offs_d[None, :] * k_dim_stride)
        v_ptrs = V_ptr + v_offset + (offs_n_curr[:, None] * v_seq_stride + offs_d[None, :] * v_dim_stride)
        
        k = tl.load(k_ptrs, mask=offs_n_curr[:, None] < N_CTX)
        v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N_CTX)
        
        # 计算注意力分数
        qk = tl.dot(q, tl.trans(k))
        
        # 应用因果掩码
        if CAUSAL:
            causal_mask = (offs_m[:, None] >= offs_n_curr[None, :])
            qk = tl.where(causal_mask, qk, -float('inf'))
        
        # 在线softmax更新
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        
        l_ij = tl.sum(tl.exp(qk - m_ij[:, None]), 1)
        l_new = alpha * l_i + beta * l_ij
        
        # 更新累加器
        acc_scale = alpha / l_new
        acc = acc * acc_scale[:, None]
        
        # 计算当前块的贡献
        p_ij = tl.exp(qk - m_new[:, None])
        acc += tl.dot(p_ij, v) * (beta / l_new)[:, None]
        
        # 更新统计量
        l_i = l_new
        m_i = m_new
    
    # 存储输出
    o_ptrs = O_ptr + o_offset + (offs_m[:, None] * o_seq_stride + offs_d[None, :] * o_dim_stride)
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < N_CTX)
    
    # 存储统计量（用于反向传播）
    l_ptrs = L_ptr + l_offset + offs_m * l_seq_stride
    m_ptrs = M_ptr + m_offset + offs_m * m_seq_stride
    tl.store(l_ptrs, l_i, mask=offs_m < N_CTX)
    tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)


@triton.jit
def _flash_attention_backward_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, M_ptr,
    dO_ptr, dQ_ptr, dK_ptr, dV_ptr,
    q_batch_stride, q_head_stride, q_seq_stride, q_dim_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_dim_stride,
    v_batch_stride, v_head_stride, v_seq_stride, v_dim_stride,
    o_batch_stride, o_head_stride, o_seq_stride, o_dim_stride,
    do_batch_stride, do_head_stride, do_seq_stride, do_dim_stride,
    dq_batch_stride, dq_head_stride, dq_seq_stride, dq_dim_stride,
    dk_batch_stride, dk_head_stride, dk_seq_stride, dk_dim_stride,
    dv_batch_stride, dv_head_stride, dv_seq_stride, dv_dim_stride,
    l_batch_stride, l_head_stride, l_seq_stride,
    m_batch_stride, m_head_stride, m_seq_stride,
    N_CTX, N_HEAD, N_DIM,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, CAUSAL: tl.constexpr,
):
    """Flash Attention反向传播内核"""
    # 获取当前块的索引
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # 计算batch和head索引
    off_b = off_hz // N_HEAD
    off_h = off_hz % N_HEAD
    
    # 指针偏移
    q_offset = off_b * q_batch_stride + off_h * q_head_stride
    k_offset = off_b * k_batch_stride + off_h * k_head_stride
    v_offset = off_b * v_batch_stride + off_h * v_head_stride
    o_offset = off_b * o_batch_stride + off_h * o_head_stride
    do_offset = off_b * do_batch_stride + off_h * do_head_stride
    dq_offset = off_b * dq_batch_stride + off_h * dq_head_stride
    l_offset = off_b * l_batch_stride + off_h * l_head_stride
    m_offset = off_b * m_batch_stride + off_h * m_head_stride
    
    # 序列索引
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # 加载Q, O, dO, L, M
    q_ptrs = Q_ptr + q_offset + (offs_m[:, None] * q_seq_stride + offs_d[None, :] * q_dim_stride)
    o_ptrs = O_ptr + o_offset + (offs_m[:, None] * o_seq_stride + offs_d[None, :] * o_dim_stride)
    do_ptrs = dO_ptr + do_offset + (offs_m[:, None] * do_seq_stride + offs_d[None, :] * do_dim_stride)
    l_ptrs = L_ptr + l_offset + offs_m * l_seq_stride
    m_ptrs = M_ptr + m_offset + offs_m * m_seq_stride
    
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX)
    o = tl.load(o_ptrs, mask=offs_m[:, None] < N_CTX)
    do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX)
    l_i = tl.load(l_ptrs, mask=offs_m < N_CTX)
    m_i = tl.load(m_ptrs, mask=offs_m < N_CTX)
    
    # 缩放因子
    scale = 1.0 / math.sqrt(BLOCK_DMODEL)
    q = q * scale
    
    # 计算 Di = sum(dO * O)
    Di = tl.sum(do * o, axis=1)
    
    # 初始化dQ累加器
    dq_acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # 遍历所有K、V块计算dQ
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n
        
        # 因果掩码检查
        if CAUSAL:
            if start_n > start_m * BLOCK_M + BLOCK_M - 1:
                break
        
        # 加载K和V
        k_ptrs = K_ptr + k_offset + (offs_n_curr[:, None] * k_seq_stride + offs_d[None, :] * k_dim_stride)
        v_ptrs = V_ptr + v_offset + (offs_n_curr[:, None] * v_seq_stride + offs_d[None, :] * v_dim_stride)
        
        k = tl.load(k_ptrs, mask=offs_n_curr[:, None] < N_CTX)
        v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N_CTX)
        
        # 重新计算注意力分数
        qk = tl.dot(q, tl.trans(k))
        
        # 应用因果掩码
        if CAUSAL:
            causal_mask = (offs_m[:, None] >= offs_n_curr[None, :])
            qk = tl.where(causal_mask, qk, -float('inf'))
        
        # 计算softmax权重
        p = tl.exp(qk - m_i[:, None]) / l_i[:, None]
        
        # 计算dS = P * (dO @ V^T - Di)
        dv_contrib = tl.dot(tl.trans(do), v)
        ds = p * (dv_contrib - Di[:, None])
        
        # 累加dQ
        dq_acc += tl.dot(ds, k) * scale
    
    # 存储dQ
    dq_ptrs = dQ_ptr + dq_offset + (offs_m[:, None] * dq_seq_stride + offs_d[None, :] * dq_dim_stride)
    tl.store(dq_ptrs, dq_acc, mask=offs_m[:, None] < N_CTX)


@triton.jit
def _flash_attention_backward_kv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, M_ptr,
    dO_ptr, dK_ptr, dV_ptr,
    q_batch_stride, q_head_stride, q_seq_stride, q_dim_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_dim_stride,
    v_batch_stride, v_head_stride, v_seq_stride, v_dim_stride,
    o_batch_stride, o_head_stride, o_seq_stride, o_dim_stride,
    do_batch_stride, do_head_stride, do_seq_stride, do_dim_stride,
    dk_batch_stride, dk_head_stride, dk_seq_stride, dk_dim_stride,
    dv_batch_stride, dv_head_stride, dv_seq_stride, dv_dim_stride,
    l_batch_stride, l_head_stride, l_seq_stride,
    m_batch_stride, m_head_stride, m_seq_stride,
    N_CTX, N_HEAD, N_DIM,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, CAUSAL: tl.constexpr,
):
    """计算dK和dV的内核"""
    # 获取当前块的索引
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # 计算batch和head索引
    off_b = off_hz // N_HEAD
    off_h = off_hz % N_HEAD
    
    # 指针偏移
    k_offset = off_b * k_batch_stride + off_h * k_head_stride
    v_offset = off_b * v_batch_stride + off_h * v_head_stride
    dk_offset = off_b * dk_batch_stride + off_h * dk_head_stride
    dv_offset = off_b * dv_batch_stride + off_h * dv_head_stride
    
    # 序列索引
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # 加载当前块的K和V
    k_ptrs = K_ptr + k_offset + (offs_n[:, None] * k_seq_stride + offs_d[None, :] * k_dim_stride)
    v_ptrs = V_ptr + v_offset + (offs_n[:, None] * v_seq_stride + offs_d[None, :] * v_dim_stride)
    
    k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX)
    
    # 初始化累加器
    dk_acc = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    
    # 缩放因子
    scale = 1.0 / math.sqrt(BLOCK_DMODEL)
    
    # 遍历所有Q块
    for start_m in range(0, N_CTX, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        
        # 因果掩码检查
        if CAUSAL:
            if start_m > start_n * BLOCK_N + BLOCK_N - 1:
                continue
        
        # 加载Q, dO, L, M, O
        q_offset = off_b * q_batch_stride + off_h * q_head_stride
        do_offset = off_b * do_batch_stride + off_h * do_head_stride
        o_offset_curr = off_b * o_batch_stride + off_h * o_head_stride
        l_offset = off_b * l_batch_stride + off_h * l_head_stride
        m_offset = off_b * m_batch_stride + off_h * m_head_stride
        
        q_ptrs = Q_ptr + q_offset + (offs_m_curr[:, None] * q_seq_stride + offs_d[None, :] * q_dim_stride)
        do_ptrs = dO_ptr + do_offset + (offs_m_curr[:, None] * do_seq_stride + offs_d[None, :] * do_dim_stride)
        o_ptrs = O_ptr + o_offset_curr + (offs_m_curr[:, None] * o_seq_stride + offs_d[None, :] * o_dim_stride)
        l_ptrs = L_ptr + l_offset + offs_m_curr * l_seq_stride
        m_ptrs = M_ptr + m_offset + offs_m_curr * m_seq_stride
        
        q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < N_CTX)
        do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < N_CTX)
        o = tl.load(o_ptrs, mask=offs_m_curr[:, None] < N_CTX)
        l_i = tl.load(l_ptrs, mask=offs_m_curr < N_CTX)
        m_i = tl.load(m_ptrs, mask=offs_m_curr < N_CTX)
        
        q = q * scale
        
        # 重新计算注意力分数
        qk = tl.dot(q, tl.trans(k))
        
        # 应用因果掩码
        if CAUSAL:
            causal_mask = (offs_m_curr[:, None] >= offs_n[None, :])
            qk = tl.where(causal_mask, qk, -float('inf'))
        
        # 计算softmax权重
        p = tl.exp(qk - m_i[:, None]) / l_i[:, None]
        
        # 累加dV = P^T @ dO
        dv_acc += tl.dot(tl.trans(p), do)
        
        # 计算Di = sum(dO * O)
        Di = tl.sum(do * o, axis=1)
        
        # 计算dS = P * (dO @ V^T - Di)
        dv_contrib = tl.dot(do, tl.trans(v))
        ds = p * (dv_contrib - Di[:, None])
        
        # 累加dK = dS^T @ Q
        dk_acc += tl.dot(tl.trans(ds), q) * scale
    
    # 存储dK和dV
    dk_ptrs = dK_ptr + dk_offset + (offs_n[:, None] * dk_seq_stride + offs_d[None, :] * dk_dim_stride)
    dv_ptrs = dV_ptr + dv_offset + (offs_n[:, None] * dv_seq_stride + offs_d[None, :] * dv_dim_stride)
    
    tl.store(dk_ptrs, dk_acc, mask=offs_n[:, None] < N_CTX)
    tl.store(dv_ptrs, dv_acc, mask=offs_n[:, None] < N_CTX)


class FlashAttentionFunction(torch.autograd.Function):
    """Flash Attention自动求导函数"""
    
    @staticmethod
    def forward(ctx, q, k, v, causal=True):
        # 检查输入形状
        batch, num_heads, seq_len, d_model = q.shape
        assert k.shape == v.shape == q.shape
        
        # 块大小配置
        BLOCK_M = min(64, seq_len)
        BLOCK_N = min(64, seq_len)
        BLOCK_DMODEL = d_model
        
        # 确保d_model是2的幂
        assert d_model in {16, 32, 64, 128, 256}, f"d_model={d_model} not supported"
        
        # 输出张量
        o = torch.empty_like(q)
        l = torch.empty((batch, num_heads, seq_len), device=q.device, dtype=torch.float32)
        m = torch.empty((batch, num_heads, seq_len), device=q.device, dtype=torch.float32)
        
        # 计算grid维度
        grid_m = triton.cdiv(seq_len, BLOCK_M)
        grid = (grid_m, batch * num_heads)
        
        # 启动前向内核
        _flash_attention_forward_kernel[(grid_m, batch * num_heads)](
            q, k, v, o, l, m,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            l.stride(0), l.stride(1), l.stride(2),
            m.stride(0), m.stride(1), m.stride(2),
            seq_len, num_heads, d_model,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
            CAUSAL=causal,
            num_warps=4,
            num_stages=3,
        )
        
        # 保存用于反向传播的张量
        ctx.save_for_backward(q, k, v, o, l, m)
        ctx.causal = causal
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_DMODEL = BLOCK_DMODEL
        
        return o
    
    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, l, m = ctx.saved_tensors
        causal = ctx.causal
        BLOCK_M = ctx.BLOCK_M
        BLOCK_N = ctx.BLOCK_N
        BLOCK_DMODEL = ctx.BLOCK_DMODEL
        
        batch, num_heads, seq_len, d_model = q.shape
        
        # 梯度张量
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        
        # 计算dQ的grid
        grid_m = triton.cdiv(seq_len, BLOCK_M)
        grid = (grid_m, batch * num_heads)
        
        # 启动dQ内核
        _flash_attention_backward_kernel[(grid_m, batch * num_heads)](
            q, k, v, o, l, m,
            grad_output, grad_q, grad_k, grad_v,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            grad_q.stride(0), grad_q.stride(1), grad_q.stride(2), grad_q.stride(3),
            grad_k.stride(0), grad_k.stride(1), grad_k.stride(2), grad_k.stride(3),
            grad_v.stride(0), grad_v.stride(1), grad_v.stride(2), grad_v.stride(3),
            l.stride(0), l.stride(1), l.stride(2),
            m.stride(0), m.stride(1), m.stride(2),
            seq_len, num_heads, d_model,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
            CAUSAL=causal,
            num_warps=4,
            num_stages=3,
        )
        
        # 计算dK, dV的grid
        grid_n = triton.cdiv(seq_len, BLOCK_N)
        grid_kv = (grid_n, batch * num_heads)
        
        # 启动dK, dV内核
        _flash_attention_backward_kv_kernel[(grid_n, batch * num_heads)](
            q, k, v, o, l, m,
            grad_output, grad_k, grad_v,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            grad_k.stride(0), grad_k.stride(1), grad_k.stride(2), grad_k.stride(3),
            grad_v.stride(0), grad_v.stride(1), grad_v.stride(2), grad_v.stride(3),
            l.stride(0), l.stride(1), l.stride(2),
            m.stride(0), m.stride(1), m.stride(2),
            seq_len, num_heads, d_model,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
            CAUSAL=causal,
            num_warps=4,
            num_stages=3,
        )
        
        return grad_q, grad_k, grad_v, None


class FlashAttention(torch.nn.Module):
    """Flash Attention层"""
    
    def __init__(self, d_model: int, num_heads: int, causal: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.causal = causal
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        # 线性投影层
        self.w_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_o = torch.nn.Linear(d_model, d_model, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            torch.nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, d_model]
            mask: 可选的注意力掩码
            
        Returns:
            输出张量 [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape
        
        # 线性投影
        q = self.w_q(x)  # [batch, seq_len, d_model]
        k = self.w_k(x)  # [batch, seq_len, d_model]
        v = self.w_v(x)  # [batch, seq_len, d_model]
        
        # 重塑为多头
        q = q.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)  # [batch, num_heads, seq_len, d_head]
        k = k.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)  # [batch, num_heads, seq_len, d_head]
        v = v.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)  # [batch, num_heads, seq_len, d_head]
        
        # Flash Attention
        attention_out = FlashAttentionFunction.apply(q, k, v, self.causal)
        
        # 重塑回原始形状
        out = attention_out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        
        # 输出投影
        result = self.w_o(out)
        
        return result


def flash_attention(q: torch.Tensor, 
                   k: torch.Tensor, 
                   v: torch.Tensor, 
                   causal: bool = True) -> torch.Tensor:
    return FlashAttentionFunction.apply(q, k, v, causal)


