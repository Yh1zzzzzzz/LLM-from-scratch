import torch

def Cross_entropy(inputs, targets):
    inputs = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    
    target_logits = inputs[torch.arange(inputs.shape[0]), targets]
    
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs), dim=-1))
    
    loss = -target_logits + log_sum_exp
    return loss.mean()


