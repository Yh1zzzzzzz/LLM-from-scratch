import torch
import torch.nn as nn
import math

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FFN, self).__init__()
        
        self.W1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b1 = nn.Parameter(torch.empty(d_ff))
        self.W2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b2 = nn.Parameter(torch.empty(d_model))
        
        self.dropout_p = dropout
        
        self._init_weights()
        
    def _init_weights(self):
        std = math.sqrt(2.0 / (self.W1.shape[0] + self.W1.shape[1]))
        nn.init.normal_(self.W1, 0.0, std)
        nn.init.zeros_(self.b1)
        
        std = math.sqrt(2.0 / (self.W2.shape[0] + self.W2.shape[1]))
        nn.init.normal_(self.W2, 0.0, std)
        nn.init.zeros_(self.b2)
        
    def relu(self, x):
        return torch.where(x > 0, x, torch.zeros_like(x))
    
    def dropout(self, x, p, training=True):
        if not training or p == 0:
            return x
            
        keep_prob = 1 - p
        mask = torch.rand_like(x) < keep_prob
        
        return x * mask.float() / keep_prob
        
    def forward(self, x):
        residual = x
        
        x = torch.matmul(x, self.W1) + self.b1
        
        x = self.relu(x)
        
        x = self.dropout(x, self.dropout_p, self.training)
        
        x = torch.matmul(x, self.W2) + self.b2
        
        return x + residual



