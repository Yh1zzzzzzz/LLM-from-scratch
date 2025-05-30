import torch
import torch.nn as nn
from torch.nn import functional as F
import einops
from einops import einsum
class SwiGLU(nn.Module):
    def __init__(self,  d_model, dff, device=torch.device("mps"), dtype=torch.float32):
        super().__init__()
        self.d_model = d_model
        self.ff = dff
        self.W1 = nn.Parameter(torch.empty(dff, d_model, device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(d_model, dff, device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(dff, d_model, device=device, dtype=dtype))
        self.device = device
        self.dtype = dtype
    @staticmethod
    def silu(x : torch.Tensor):
        return x * torch.sigmoid(x) #use sigmoid for nunmerical stability
    def forward(self, x):
        print(x.shape)
        print(self.W1.shape)
        x = x.to(self.device).to(self.dtype)
        return (self.silu(x @ self.W1.T) * (x @ self.W3.T)) @ self.W2.T
    
