import torch
import torch.nn as nn

from jaxtyping import Float, Int
from torch import Tensor

from .embedding import Embedding, PosEmbedding, Unembedding
from .attention import Attention
from .layernorm import LayerNorm
from .mlp import MLP


class TransformerBlock(nn.Module):
    """Implmentation of Transformer block for transformer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln1 = LayerNorm(config)
        self.attn = Attention(config)
        self.ln2 = LayerNorm(config)
        self.mlp = MLP(config)

    def forward(
        self, resid_pre: Float[torch.Tensor, "batch seq_len d_model"]
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        """
        Forward pass for transformer block.
        """
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post


class Transformer(nn.Module):
    """Implementation of Transformer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = Embedding(config)
        self.pos_embed = PosEmbedding(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_final = LayerNorm(config)
        self.unembed = Unembedding(config)

    def forward(self, tokens: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len d_vocab"]:
        """
        Forward pass for transformer.
        """
        residual = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        residual = self.ln_final(residual)
        return self.unembed(residual)
