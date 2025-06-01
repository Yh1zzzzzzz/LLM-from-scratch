import torch
from Architecture.model import BasicsTransformerLM

device = 'cpu'  # Use CPU to avoid device issues

config = {
    'vocab_size': 100,
    'context_length': 32,
    'd_model': 64,
    'num_layers': 2,
    'num_heads': 4,
    'd_ff': 128,
    'rope_theta': 10000.0
}

model = BasicsTransformerLM(**config).to(device)
x = torch.randint(0, config['vocab_size'], (2, 16)).to(device)

with torch.no_grad():
    output = model(x)
    print(f"Success! Output shape: {output.shape}")
    gen_tokens = model.generate(x[0:1], max_new_tokens=3)
    print(f"Generation success! Shape: {gen_tokens.shape}")
