import torch
import torch.nn as nn
from math import sin, cos

class RoPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=torch.device("mps")) :
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.register_buffer("Rotation_Matrix" ,torch.empty(max_seq_len, d_k, d_k, device=device))
        for i in range(max_seq_len):
            for j in range(d_k // 2):
                theta_i = i / theta ** (2 * j / d_k)
                self.Rotation_Matrix[i, 2*j : 2*j+2, 2*j : 2*j+2] = \
                torch.tensor([[cos(theta_i), -sin(theta_i)],[sin(theta_i), cos(theta_i)]], device=device) 

    
    def forward(self, x:torch.Tensor, token_positon:torch.Tensor):
        x = x.to(torch.device("mps"))
        print(x.shape)
        print(token_positon.shape)
        seq_len = token_positon.shape[-1]
        for i in range(seq_len):
            #[batch , positon]
            Batch_pos = token_positon[...,i]# [batch]
            sub_MAT = self.Rotation_Matrix[Batch_pos, :, :]  # [batch, d_k, d_k
            for j in range(self.d_k // 2):
                x[..., i, 2*j : 2*j+2]  = torch.mm(x[..., i, 2*j : 2*j+2] , sub_MAT[...,2*j : 2*j+2, 2*j : 2*j+2].T)
        return x
    
    