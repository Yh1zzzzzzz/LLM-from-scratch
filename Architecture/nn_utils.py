import torch
import torch.nn.functional as F

def softmax(x, dim=-1):
    """Softmax function"""
    return F.softmax(x, dim=dim)
