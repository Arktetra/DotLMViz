import einops
import torch
import torch.nn as nn

from jaxtyping import Float

from DoTLMViz.activations import gelu_new
from DoTLMViz.core import Ckpt


class MLP(nn.Module):
    """A standard neural network, with a singular hidden layer and a non-linear
    activation function."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W_in = nn.Parameter(torch.empty((config.d_model, config.d_mlp)))
        self.b_in = nn.Parameter(torch.zeros((config.d_mlp)))
        self.W_out = nn.Parameter(torch.empty((config.d_mlp, config.d_model)))
        self.b_out = nn.Parameter(torch.zeros((config.d_model)))
        nn.init.normal_(self.W_in, std=config.init_range)
        nn.init.normal_(self.W_out, std=config.init_range)
        self.ckpt_pre = Ckpt()
        self.ckpt_post = Ckpt()
        self.ckpt_mlp_out = Ckpt()

    def forward(
        self, residual: Float[torch.Tensor, "batch seq_len d_model"]
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        """Forward pass for MLP"""
        residual = (
            einops.einsum(residual, self.W_in, "batch seq_len d_model, d_model d_mlp -> batch seq_len d_mlp")
            + self.b_in
        )
        self.ckpt_pre(residual)
        residual = gelu_new(residual)
        self.ckpt_post(residual)
        residual = (
            einops.einsum(residual, self.W_out, "batch seq_len d_mlp, d_mlp d_model -> batch seq_len d_model")
            + self.b_out
        )
        self.ckpt_mlp_out(residual)
        return residual
