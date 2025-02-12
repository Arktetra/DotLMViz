import numpy as np
import torch
import torch.distributions as D

from jaxtyping import Int, Float
from torch import Tensor


class TransformerSampler:
    """
    A class for sampler.
    """

    @torch.inference_mode
    @staticmethod
    def sample_next_token(
        input_ids: Int[Tensor, " seq_len"],
        logits: Float[Tensor, " d_vocab"],
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        frequency_penalty: float = 0.0,
        seed: int = None,
    ) -> int:
        """
        Samples the next token.

        Args:
            input_ids (Int[Tensor, " seq_len"]): token ids.
            logits (Float[Tensor, " d_vocab"]): logits from transformer.
            temperature (float, optional): scaling value. Defaults to 1.0.
            top_k (int, optional): number of logits to sample from. Defaults to 0.
            top_p (float, optional): cumulative probability value. Defaults to 0.0.
            frequency_penalty (float, optional): frequency penalty value. Defaults to 0.0.
            seed (int, optional): seed for determinism. Defaults to None.

        Returns:
            int: token id of the next token.
        """

        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert 0 <= top_p <= 1.0, "top-p must be in [0, 1]"
        assert 0 <= top_k, "top-k must be non-negative"
        assert not (top_p != 0 and top_k != 0), "choose either top_p or top_k"

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)

        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(input_ids, logits, frequency_penalty)

        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)

        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)

        return TransformerSampler.sample_basic(logits)

    @staticmethod
    def greedy_search(logits: Float[Tensor, " d_vocab"]) -> int:
        """
        Returns the most likely token.

        Args:
            logits (Float[Tensor, " d_vocab"]): logits from transformer.

        Returns:
            int: token id of the next token.
        """
        return logits.argmax().item()

    @staticmethod
    def apply_temperature(logits: Float[Tensor, " d_vocab"], temperature: float) -> Float[Tensor, " d_vocab"]:
        """
        Applies temperature scaling to the logits.

        Argss:
            logits (Float[Tensor, " d_vocab"]): logits from transformer.
            temperature (float): value used for scaling the logits.

        Returns:
            Float[Tensor, " d_vocab"]: scaled logits.
        """
        return logits / temperature

    @staticmethod
    def apply_frequency_penalty(
        input_ids: Int[Tensor, " seq_len"], logits: Float[Tensor, " d_vocab"], freq_penalty: float
    ) -> Float[Tensor, " d_vocab"]:
        """
        Applies a frequency penalty to the logits.

        Args:
            input_ids (Int[Tensor, " seq_len"]): token ids.
            logits (Float[Tensor, " d_vocab"]): logits from transformer.
            freq_penalty (float): frequency penalty that is applied to the logits.

        Returns:
            Float[Tensor, " d_vocab"]: penalized logits.
        """
        (vocab_size,) = logits.shape
        freq = torch.bincount(input_ids, minlength=vocab_size)
        return logits - freq_penalty * freq

    @staticmethod
    def sample_basic(logits: Float[Tensor, " d_vocab"]) -> int:
        """Samples from the distribution defined by the logits.

        Args:
            logits (Float[Tensor, &quot; d_vocab&quot;]): logits.

        Returns:
            int: token id.
        """
        dist = D.Categorical(logits=logits)
        sampled_token = dist.sample().item()
        return sampled_token

    @staticmethod
    def sample_top_k(logits: Float[Tensor, " d_vocab"], k: int) -> int:
        """
        Samples from the top k most likely tokens.

        Args:
            logits (Float[Tensor, &quot; d_vocab&quot;]): logits_
            k (int): number of likely tokens to sample from.

        Returns:
            int: token id of the next token.
        """
        top_logits, top_idxs = torch.topk(logits, k)
        idx = D.Categorical(logits=top_logits).sample()
        return top_idxs[idx].item()

    @staticmethod
    def sample_top_p(logits: Float[Tensor, " d_vocab"], top_p: float, min_tokens_to_keep: int = 1) -> int:
        """
        Samples from the most likely tokens which make up at
        least p cumulative probaiblity.

        Args:
            logits (Float[Tensor, &quot; d_vocab&quot;]): logits from transformer.
            top_p (float): cumulative probability.
            min_tokens_to_keep (int, optional): minimum number of tokens to keep. Defaults to 1.

        Returns:
            int: token id of the next token.
        """

        sorted_logits, idxs = torch.sort(logits, dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        cum_probs = probs.cumsum(dim=-1)
        n_keep = torch.searchsorted(cum_probs, top_p, right=True).item() + 1
        n_keep = max(n_keep, min_tokens_to_keep)
        keep_idx = idxs[:n_keep]
        keep_logits = logits[:n_keep]
        idx = D.Categorical(logits=keep_logits).sample()
        return keep_idx[idx].item()
