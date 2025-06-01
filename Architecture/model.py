from __future__ import annotations

import functools
import json
import logging
import math
import os
from einops import rearrange, einsum

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int

from .FFN import FFN
from .RMSNorm import RMSNorm  
from .MultiHeadAtten import MultiheadAttention, RoPE
from .nn_utils import softmax

logger = logging.getLogger(__name__)


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        std = math.sqrt(2 / (d_in + d_out))
        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(d_out, d_in), std=std, a=-3*std, b=3*std),
            requires_grad=True
        )

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    
    def extra_repr(self):
        return f"d_out={self.weight.shape[0]}, d_in={self.weight.shape[1]}"


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        std = 1.0
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(vocab_size, d_model), std=std, a=-3 * std, b=3 * std),
            requires_grad=True
        )
    
    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids, :]
    
    def extra_repr(self):
        return f"vocab_size={self.weight.shape[0]}, d={self.weight.shape[1]}"


class BasicsTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        self.config = {
            k: v for k, v in locals().items() if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        
        self.token_embeddings = Embedding(vocab_size, d_model)
        
        d_head = d_model // num_heads
        self.positional_encoder = RoPE(
            theta=rope_theta,
            d_k=d_head,
            max_seq_len=context_length
        )
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    positional_encoder=self.positional_encoder,
                )
                for _ in range(num_layers)
            ]
        )
        
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

        logger.info(f"number of non-embedding parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.lm_head.weight.numel()
        return n_params

    def forward(self, x: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length vocab_size"]:
        _, sequence_length = x.size()
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        original_sequence_length = x.size(-1)
        for _ in range(max_new_tokens):
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x
            logits = self.forward(x)
            next_token_logits = logits[:, -1]
            temperature_scaled_next_token_logits = next_token_logits / temperature
            if top_k:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, temperature_scaled_next_token_logits.size(-1)),
                )
                threshold = topk_values[:, -1]
                topk_mask = temperature_scaled_next_token_logits < threshold
                # 避免就地操作，创建新张量
                temperature_scaled_next_token_logits = temperature_scaled_next_token_logits.masked_fill(topk_mask, float("-inf"))
            next_token_probabilities = softmax(temperature_scaled_next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probabilities, 1)
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)
        new_token_ids = x[:, original_sequence_length:]
        return new_token_ids

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path) as f:
            config = json.load(f)
        model = cls(**config)
        weights_path = os.path.join(pretrained_model_path, "model.pt")
        state_dict = torch.load(weights_path)

        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        positional_encoder: RoPE,
    ):
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            positional_encoder=positional_encoder,
        )
        self.ffn = FFN(d_model=d_model, d_ff=d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor):
        x_attn = self.attn(self.ln1(x))
        attn_sublayer_output = x + x_attn
        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        return ffn_sublayer_output


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RoPE,
    ):
        super().__init__()
        self.attention = MultiheadAttention(d_model, num_heads)
        self.positional_encoder = positional_encoder

    def forward(self, x: Float[Tensor, " ... seq d_k"], token_positions: Int[Tensor, " ... seq"] | None = None) -> Float[Tensor, " ... seq d_v"]:
        return self.attention(x)


