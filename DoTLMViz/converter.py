import einops

from torch import Tensor
from typing import Dict

from .transformers import Config


class Converter:
    """
    Converter for converting the state dict containing merged weights to the ones
    needed by our transformer models.

    Example:
    -------

    >>> from DoTLMViz.transformers import GPT2SmallConfig
    >>> from DoTLMViz.converter import Converter
    >>>
    >>> state_dict = Converter(model_name = "gpt2-small", config = GPT2SmallConfig).convert()
    """

    def __init__(self, model_name: str, config: Config):
        self.model_name = model_name
        self.config = config

    def convert(self) -> Dict[str, Tensor]:
        """
        Performs the conversion.

        Returns:
            Dict[str, Tensor]: state dict for the model specified by `config`.
        """
        if self.model_name == "gpt2-small":
            return convert_gpt2(self.config)


def convert_gpt2(config: Config) -> Dict[str, Tensor]:
    """
    Converts the state dict from gpt2 to have a state dict for a model as
    specified by config.

    Args:
        config (Config): model config

    Returns:
        Dict[str, Tensor]: state dict for the model specified by `config`.
    """
    from transformers import GPT2Model

    gpt2 = GPT2Model.from_pretrained("gpt2")

    state_dict = {}

    state_dict["embed.W_E"] = gpt2.wte.weight
    state_dict["pos_embed.W_pos"] = gpt2.wpe.weight

    for i in range(config.n_layers):
        state_dict[f"blocks.{i}.ln1.w"] = gpt2.h[i].ln_1.weight
        state_dict[f"blocks.{i}.ln1.b"] = gpt2.h[i].ln_1.bias

        W_qkv = einops.rearrange(
            gpt2.h[i].attn.c_attn.weight,
            "d_model (n_states n_heads d_head) -> n_states n_heads d_model d_head",
            n_states=3,
            n_heads=config.n_heads,
        )

        W_o = einops.rearrange(
            gpt2.h[i].attn.c_proj.weight, "(n_heads d_head) d_model -> n_heads d_head d_model", n_heads=config.n_heads
        )

        state_dict[f"blocks.{i}.attn.W_Q"] = W_qkv[0]
        state_dict[f"blocks.{i}.attn.W_K"] = W_qkv[1]
        state_dict[f"blocks.{i}.attn.W_V"] = W_qkv[2]
        state_dict[f"blocks.{i}.attn.W_O"] = W_o

        b_qkv = einops.rearrange(
            gpt2.h[i].attn.c_attn.bias,
            "(n_states n_heads d_head) -> n_states n_heads d_head",
            n_states=3,
            n_heads=config.n_heads,
        )

        b_o = gpt2.h[i].attn.c_proj.bias

        state_dict[f"blocks.{i}.attn.b_Q"] = b_qkv[0]
        state_dict[f"blocks.{i}.attn.b_K"] = b_qkv[1]
        state_dict[f"blocks.{i}.attn.b_V"] = b_qkv[2]
        state_dict[f"blocks.{i}.attn.b_O"] = b_o

        state_dict[f"blocks.{i}.ln2.w"] = gpt2.h[i].ln_2.weight
        state_dict[f"blocks.{i}.ln2.b"] = gpt2.h[i].ln_2.bias

        state_dict[f"blocks.{i}.mlp.W_in"] = gpt2.h[i].mlp.c_fc.weight
        state_dict[f"blocks.{i}.mlp.b_in"] = gpt2.h[i].mlp.c_fc.bias

        state_dict[f"blocks.{i}.mlp.W_out"] = gpt2.h[i].mlp.c_proj.weight
        state_dict[f"blocks.{i}.mlp.b_out"] = gpt2.h[i].mlp.c_proj.bias

    state_dict["ln_final.w"] = gpt2.ln_f.weight
    state_dict["ln_final.b"] = gpt2.ln_f.bias

    state_dict["unembed.W_U"] = gpt2.wte.weight.T

    return state_dict
