import torch
import torch.nn as nn

from functools import partial
from jaxtyping import Float, Int
from typing import Dict, List, Tuple
from torch import Tensor
from transformers import GPT2Model

from .core.hook import Hooks
from .transformers import Transformer, TransformerBlock, GPT2SmallConfig


class HookedTransformer:
    """Hook a Transformer model to cache the intermediate outputs of each layer in the model."""

    def __init__(self, module: Transformer):
        self.module = module
        self.cache = {}

    @staticmethod
    def from_pretrained(model_name: str):
        if model_name == "gpt2-small":
            return HookedTransformer.__load_gpt2_small()

    @staticmethod
    def __load_gpt2_small():
        def __split_states(x, n):
            *start, m = x.shape
            return x.reshape(start + [n, m // n])

        def __split_heads(x, n_heads):
            return torch.transpose(__split_states(x, n_heads), dim0=-1, dim1=-2)

        new_model = Transformer(GPT2SmallConfig)
        pretrained_model = GPT2Model.from_pretrained("gpt2")

        new_state_dict = new_model.state_dict()
        new_state_keys = list(new_state_dict.keys())
        pre_state_dict = pretrained_model.state_dict()
        pre_state_keys = list(pre_state_dict.keys())

        new_iter = 0
        pre_iter = 0

        for i in range(len(new_state_dict)):
            if pre_iter == len(pre_state_keys):
                break

            new_key = new_state_keys[new_iter]
            pre_key = pre_state_keys[pre_iter]

            if "attn" in new_key:
                c_attn_weight = pre_state_dict[pre_state_keys[pre_iter]]
                c_attn_bias = pre_state_dict[pre_state_keys[pre_iter + 1]]
                c_proj_weight = pre_state_dict[pre_state_keys[pre_iter + 2]]
                c_proj_bias = pre_state_dict[pre_state_keys[pre_iter + 3]]

                W_Q, W_K, W_V = map(
                    partial(__split_heads, n_heads=new_model.config.n_heads),
                    torch.split(c_attn_weight, 2304 // 3, dim=-1),
                )
                W_Q, W_K, W_V = map(partial(torch.permute, dims=(2, 0, 1)), [W_Q, W_K, W_V])
                b_Q, b_K, b_V = map(
                    partial(__split_heads, n_heads=new_model.config.n_heads),
                    torch.split(c_attn_bias, 2304 // 3, dim=-1),
                )
                b_Q, b_K, b_V = map(partial(torch.permute, dims=(1, 0)), [b_Q, b_K, b_V])
                W_O = __split_states(c_proj_weight.permute(1, 0), 12).permute(1, 2, 0)
                b_O = c_proj_bias

                new_state_dict[new_state_keys[new_iter]] = W_Q
                new_state_dict[new_state_keys[new_iter + 1]] = W_K
                new_state_dict[new_state_keys[new_iter + 2]] = W_V
                new_state_dict[new_state_keys[new_iter + 3]] = W_O
                new_state_dict[new_state_keys[new_iter + 4]] = b_Q
                new_state_dict[new_state_keys[new_iter + 5]] = b_K
                new_state_dict[new_state_keys[new_iter + 6]] = b_V
                new_state_dict[new_state_keys[new_iter + 7]] = b_O
                new_iter += 9  # 8 ( + 1 for skipping IGNORE)
                pre_iter += 4
            elif "unembed" in new_key:
                new_state_dict[new_key] = pre_state_dict[pre_key].permute(1, 0)
                new_iter += 2
            else:
                new_state_dict[new_key] = pre_state_dict[pre_key]
                new_iter += 1
                pre_iter += 1
        new_state_dict["unembed.W_U"] = pre_state_dict["wte.weight"].permute(1, 0)

        new_model.load_state_dict(new_state_dict, strict=True)

        return HookedTransformer(new_model)

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
