import einops

import torch
import torch.nn as nn

from jaxtyping import Float, Int
from torch import Tensor


class Embedding(nn.Module):
    """The token embedding layer for mapping input tokens into vectors."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W_E = nn.Parameter(torch.empty((config.d_vocab, config.d_model)))
        nn.init.normal_(self.W_E, std=self.config.init_range)

    def forward(self, tokens: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len d_model"]:
        """The token ids are used to access the lookup table represented by the
        matrix $W_E$.

        Args:
            tokens (Int[Tensor, "batch seq_len"]): the token ids for the each
            token in the sequence.

        Returns:
            Float[Tensor, "batch seq_len d_model"]: the corresponding
            embedding vectors for each token in the sequence.
        """
        return self.W_E[tokens]


class PosEmbedding(nn.Module):
    """The position embedding layer for mapping the index of the position of
    each token to a residual stream vector."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W_pos = nn.Parameter(torch.empty((config.n_ctx, config.d_model)))
        nn.init.normal_(self.W_pos, std=config.init_range)

    def forward(self, tokens: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len d_model"]:
        batch, seq_len = tokens.shape
        return einops.repeat(self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch)


class Unembedding(nn.Module):
    """Projects the output at position $N$ from the final transformer block $L$
    to logit vectors."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W_U = nn.Parameter(torch.empty((config.d_model, config.d_vocab)))
        self.b_U = nn.Parameter(torch.zeros((config.d_vocab), requires_grad=False))
        nn.init.normal_(self.W_U, std=config.init_range)

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch seq_len d_model"]
    ) -> Int[Tensor, "batch seq_len d_vocab"]:
        return (
            einops.einsum(
                normalized_resid_final,
                self.W_U,
                "batch seq_len d_model, d_model vocab_size -> batch seq_len vocab_size",
            )
            + self.b_U
        )
