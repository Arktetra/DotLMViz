import torch
import torch.nn as nn

from functools import partial
from jaxtyping import Float, Int
from typing import Dict, List, Tuple
from torch import Tensor

from .converter import Converter

from .core.hook import Hooks
from .transformers import Config, Embedding, GPT2SmallConfig, LayerNorm, PosEmbedding, TransformerBlock, Unembedding


class HookedTransformer(nn.Module):
    """Hook a Transformer model to cache the intermediate outputs of each layer in the model."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed = Embedding(config)
        self.pos_embed = PosEmbedding(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_final = LayerNorm(config)
        self.unembed = Unembedding(config)

    def forward(self, tokens: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len d_vocab"]:
        residual = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        residual = self.ln_final(residual)
        return self.unembed(residual)

    @classmethod
    def from_pretrained(cls, model_name: str, config: Config = None, device=None) -> "HookedTransformer":
        """Create a hooked transformer from a pretrained model.

        Model names that are currently supported:

        - "gpt2-small"

        Args:
            model_name (str): name of the model.
            config (Config, optional): configuration of the transformer. Defaults to None.

        Returns:
            HookedTransformer: An instance of the `HookedTransformer` created from the pretrained model.
        """
        if model_name == "gpt2-small":
            config = GPT2SmallConfig

        state_dict = Converter(model_name, config).convert()

        model = cls(config)

        model.load_state_dict(state_dict, strict=False)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        return model.to(device)

    def run_with_cache(
        self, tokens: Int[Tensor, "batch seq_len"]
    ) -> Tuple[Float[Tensor, "batch seq_len d_vocab"], Dict[str, Tensor]]:
        """Runs the model and returns the output of the model along with the cache of the
        output of each layer in the model.
        """
        modules, module_name_pairs = self.get_all_sub_modules()

        self.cache = {}
        self.hooks = Hooks(modules, partial(self._hookfunc, module_name_pairs=module_name_pairs))

        output = self(tokens)

        self.hooks.remove()

        return output, self.cache

    def _hookfunc(self, *args, **kwargs):
        self.cache_module(*args, **kwargs)

    def cache_module(self, hook, module, input, output, module_name_pairs):
        """
        The hook function that is used to cache the output of each layer in the model.
        """
        name = module_name_pairs[module]
        self.cache[name] = output

    def get_all_sub_modules(self) -> Tuple[List[nn.Module], Dict[nn.Module, str]]:
        """
        Returns a tuple of list of all sub modules and their mapping to their names in the
        model.
        """
        modules = []
        module_name_pairs = {}

        stack = [(self, "")]

        while stack:
            current_module, parent_name = stack.pop()

            for name, sub_module in current_module._modules.items():
                full_name = f"{parent_name}.{name}" if parent_name else name

                if isinstance(sub_module, (nn.ModuleList, TransformerBlock)):
                    stack.append((sub_module, full_name))
                else:
                    modules.append(sub_module)
                    module_name_pairs.update({sub_module: full_name})

        return modules, module_name_pairs
