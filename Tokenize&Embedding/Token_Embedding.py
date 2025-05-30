import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=torch.device("mps"), dtype=torch.float32):
        super().__init__()
        self.number_of_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_Matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding_Matrix, mean=0.0, std= 1, a = -3, b = 3)  # Initialize with truncated normal distribution
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input : [batch_size, sequence_length]

        output : [batch_size, sequence_length, embedding_dim]
        """
        return self.embedding_Matrix[input_ids]  # Use advanced indexing to get the embeddings for the input IDs