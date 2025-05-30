import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=torch.device("mps"), dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gi = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        
        # 确保输入在正确设备上
        x = x.to(device=self.gi.device, dtype=torch.float32)
        
        # 计算 RMS：沿最后一个维度计算均方根
        # x.shape = [batch_size, seq_len, d_model]
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        
        # 归一化并应用可学习参数
        result = (x / rms) * self.gi
        
        # 转回原始数据类型
        return result.to(dtype=in_type)