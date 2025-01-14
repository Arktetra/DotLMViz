import torch.nn as nn

from torch import Tensor


class Ckpt(nn.Module):
    """
    A module for checkpointing the outputs of other modules.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Simply return the input tensor without any modification because we are only
        using it to checkpoint a model.

        Args:
            x (Tensor): the output of a previous layer, which is the input for this
            module.

        Returns:
            Tensor: the input tensor without any modification.
        """
        return x


class ActivationCkpts(dict):
    """
    A dict mapping the activation names to the corresponding tensors. The mapping
    are created wherever `Ckpt` is used for creating the checkpoints.
    """

    def __init__(self, *args, **kwargs):
        super(ActivationCkpts, self).__init__(*args, **kwargs)

    def get(self, act_name: str, layer_name: str = None, block: int = None) -> Tensor:
        """method for getting an item from the dict containing activation checkpoints
        using `act_name`, `layer_name`, and `block`.

        Args:
            act_name (str): the name of the activation checkpoint.
            layer_name (str, optional): the name of the layer inside which the
            `act_name` lies. Defaults to None.
            block (int, optional): the block number inside which the `layer_name`
            lies. Defaults to None.

        Returns:
            Tensor: the activation tensor.
        """
        key = f"blocks.{block}." if block is not None else ""
        key = key + f"{layer_name}." if layer_name else key
        key = key + f"ckpt_{act_name}"

        return self[key]
