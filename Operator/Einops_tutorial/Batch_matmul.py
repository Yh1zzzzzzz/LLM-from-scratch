import einops
import torch

def Batch_matmul(a, b):
    # Create two random tensors
    
    c = einops.einsum(a, b, 'b i j, b j k -> b i k')
    return c
if __name__ == "__main__":
    A = torch.randn(2, 3, 4).to("mps")
    B = torch.randn(2, 4, 5).to("mps")
    C = Batch_matmul(A, B)

    Ground_truth = torch.matmul(A, B)
    print(C)
    print(Ground_truth)
    print(torch.allclose(C, Ground_truth))

