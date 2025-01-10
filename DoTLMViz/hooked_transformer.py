import torch.nn as nn

from functools import partial
from jaxtyping import Float, Int
from typing import Dict, List, Tuple
from torch import Tensor

from .core.hook import Hooks
from .transformers import Transformer, TransformerBlock


class HookedTransformer:
    """Hook a Transformer model to cache the intermediate outputs of each layer in the model."""

    def __init__(self, module: Transformer):
        self.module = module
        self.cache = {}

    def run_with_cache(
        self, tokens: Int[Tensor, "batch seq_len"]
    ) -> Tuple[Float[Tensor, "batch seq_len d_vocab"], Dict[str, Tensor]]:
        """Runs the model and returns the output of the model along with the cache of the
        output of each layer in the model.
        """
        modules, module_name_pairs = self.get_all_sub_modules()

        self.hooks = Hooks(modules, partial(self._hookfunc, module_name_pairs=module_name_pairs))

        output = self.module(tokens)

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

        stack = [(self.module, "")]

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
