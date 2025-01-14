import torch
import torch.nn as nn

from functools import partial
from jaxtyping import Float, Int
from typing import Dict, List, Tuple
from torch import Tensor

from .converter import Converter

from .core.hook import Hooks
from .core import Ckpt, ActivationCkpts
from .transformers import Config, Embedding, GPT2SmallConfig, LayerNorm, PosEmbedding, TransformerBlock, Unembedding


class CkptedTransformer(nn.Module):
    """
    Hook a Transformer model to add checkpoints for the intermediate outputs of
    each layer in the model."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed = Embedding(config)
        self.pos_embed = PosEmbedding(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_final = LayerNorm(config)
        self.unembed = Unembedding(config)

        # for checkpointing
        self.ckpt_embed = Ckpt()
        self.ckpt_pos_embed = Ckpt()

    def forward(self, tokens: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len d_vocab"]:
        embed = self.embed(tokens)
        self.ckpt_embed(embed)  # checkpoint token embed
        pos_embed = self.pos_embed(tokens)
        self.ckpt_pos_embed(pos_embed)  # checkpoint position embed
        residual = self.embed(tokens) + self.pos_embed(tokens)  # this will be checkpointed inside the transformer block
        for block in self.blocks:
            residual = block(residual)
        residual = self.ln_final(residual)
        return self.unembed(residual)

    @classmethod
    def from_pretrained(cls, model_name: str, config: Config = None, device=None) -> "CkptedTransformer":
        """
        Create a hooked transformer from a pretrained model.

        Model names that are currently supported:

        - "gpt2-small"

        Args:
            model_name (str): name of the model.
            config (Config, optional): configuration of the transformer. Defaults to None.

        Returns:
            CkptedTransformer: An instance of the `CkptedTransformer` created from the pretrained model.
        """
        if model_name == "gpt2-small":
            config = GPT2SmallConfig

        state_dict = Converter(model_name, config).convert()

        model = cls(config)

        model.load_state_dict(state_dict, strict=False)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        return model.to(device)

    def run_with_ckpts(
        self, tokens: Int[Tensor, "batch seq_len"]
    ) -> Tuple[Float[Tensor, "batch seq_len d_vocab"], Dict[str, Tensor]]:
        """
        Runs the model and returns the output of the model along with the checkpoints of the
        output of each layer in the model.
        """
        modules, module_name_pairs = self.get_all_checkpoints()

        self.ckpts = ActivationCkpts()
        self.hooks = Hooks(modules, partial(self._hookfunc, module_name_pairs=module_name_pairs))

        output = self(tokens)

        self.hooks.remove()

        return output, self.ckpts

    def _hookfunc(self, *args, **kwargs):
        self.ckpt_module(*args, **kwargs)

    def ckpt_module(self, hook, module, input, output, module_name_pairs):
        """
        The hook function that is used to checkpoint the output of each layer in the model.
        """
        name = module_name_pairs[module]
        self.ckpts[name] = output

    def get_all_checkpoints(self) -> Tuple[List[nn.Module], Dict[nn.Module, str]]:
        """
        Returns a tuple of list of all checkpoints and their mapping to their names in the
        model.
        """
        modules = []
        module_name_pairs = {}

        stack = [(self, "")]

        while stack:
            current_module, parent_name = stack.pop()

            for name, sub_module in current_module._modules.items():
                full_name = f"{parent_name}.{name}" if parent_name else name

                if not isinstance(sub_module, Ckpt):
                    stack.append((sub_module, full_name))
                else:
                    modules.append(sub_module)
                    module_name_pairs.update({sub_module: full_name})

        return modules, module_name_pairs
