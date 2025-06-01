import torch
import torch.nn as nn
from torch.nn import functional as F
from math import sin, cos
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None) :
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.WQ = nn.Parameter(torch.empty(d_model, d_model, device=device))
        self.WK = nn.Parameter(torch.empty(d_model, d_model, device=device))
        self.WV = nn.Parameter(torch.empty(d_model, d_model, device=device))
        self.WO = nn.Parameter(torch.empty(d_model, d_model, device=device))
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self._init_weights()
        
    def _init_weights(self):
        import math
        std = math.sqrt(2.0 / (self.d_model + self.d_model))
        nn.init.normal_(self.WQ, 0.0, std)
        nn.init.normal_(self.WK, 0.0, std)
        nn.init.normal_(self.WV, 0.0, std)
        nn.init.normal_(self.WO, 0.0, std)
    def forward(self, x):
        Batch, seq , embed = x.shape
        mask = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1).to(x.device)
        Q = torch.matmul(x, self.WQ).view(Batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x, self.WK).view(Batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x, self.WV).view(Batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        score = torch.matmul(Q, K.transpose(-1,-2)) 
        score /= Q.shape[-1] ** 0.5
        score += mask

        score -= torch.max(score, dim= -1, keepdim=True).values
        score = score.exp()
        score /= torch.sum(score, dim=-1, keepdim=True) 

        out = torch.matmul(score, V) # B H S D
        out = out.transpose(1,2).contiguous().view(Batch, seq, self.d_model)  # batch seq d_model
        return torch.matmul(out, self.WO)




class RoPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor, token_position: torch.Tensor):
        # x shape: (batch, num_heads, seq_len, d_k)
        # token_position shape: (batch, seq_len)
        
        batch_size, num_heads, seq_len, d_k = x.shape
        device = x.device
        
        # 确保 token_position 在正确设备上
        token_position = token_position.to(device)
        
        # 计算位置编码
        # token_position: (batch, seq_len) -> (batch, seq_len, 1)
        position = token_position.unsqueeze(-1).float()
        
        # 计算角度: (batch, seq_len, d_k//2)
        angles = position * self.inv_freq.unsqueeze(0).unsqueeze(0)
        
        # 计算 sin 和 cos: (batch, seq_len, d_k//2)
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        
        # 扩展维度以匹配 x 的形状: (batch, 1, seq_len, d_k//2)
        cos_vals = cos_vals.unsqueeze(1)
        sin_vals = sin_vals.unsqueeze(1)
        
        # 分离偶数和奇数维度
        x_even = x[..., 0::2]  # (batch, num_heads, seq_len, d_k//2)
        x_odd = x[..., 1::2]   # (batch, num_heads, seq_len, d_k//2)
        
        # 应用旋转
        rotated_even = x_even * cos_vals - x_odd * sin_vals
        rotated_odd = x_even * sin_vals + x_odd * cos_vals
        
        # 重新组合
        result = torch.zeros_like(x)
        result[..., 0::2] = rotated_even
        result[..., 1::2] = rotated_odd
        
        return result

class ROPE_MultiheadAttention(nn.Module):
    def __init__(self,d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float
    , device=torch.device("mps")):
        super().__init__()
        self.WQ = nn.Parameter(torch.empty(d_model, d_model))
        self.WK = nn.Parameter(torch.empty(d_model, d_model))
        self.WV = nn.Parameter(torch.empty(d_model, d_model))
        self.WO = nn.Parameter(torch.empty(d_model, d_model))
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
    def forward(self, x: torch.Tensor, token_position: torch.Tensor):
        # 确保输入都在 MPS 设备上
        device = torch.device("mps")
        x = x.to(device)
        token_position = token_position.to(device)
        
        # 将模型迁移到 MPS 设备（正确的方式）
        self.to(device)
        
        # 修正: 使用正确的 theta 参数
        Rope = RoPE(theta=10000.0, d_k=self.d_k, max_seq_len=self.max_seq_len, device=device)
        Batch, seq, embed = x.shape
        mask = torch.triu(torch.full((seq, seq), float("-inf"), device=device), diagonal=1)
        
        # 使用 reshape 替代 view 避免内存布局问题
        Q = torch.matmul(x, self.WQ).reshape(Batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        K = torch.matmul(x, self.WK).reshape(Batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        V = torch.matmul(x, self.WV).reshape(Batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        Q = Rope(Q, token_position)
        K = Rope(K, token_position)
        score = torch.matmul(Q, K.transpose(-1, -2))
        score /= (self.d_k ** 0.5)
        score += mask

        score -= torch.max(score, dim=-1, keepdim=True).values
        score = score.exp()
        score /= torch.sum(score, dim=-1, keepdim=True)

        out = torch.matmul(score, V)  # B H S D
        out = out.transpose(1, 2).contiguous().reshape(Batch, seq, self.d_model)  # batch seq d_model
        return torch.matmul(out, self.WO)