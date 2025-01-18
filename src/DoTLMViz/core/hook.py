import torch.nn as nn

from functools import partial
from typing import Callable, List


class Hook:
    """
    A wrapper class for managing PyTorch hook.
    """

    def __init__(self, module: nn.Module, fn: Callable):
        """
        Creates an instance of `Hook` for the module.

        Args:
            module (nn.Module): the module.
            fn (Callable): the hook function.
        """
        self.hook = module.register_forward_hook(partial(fn, self))

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


class Hooks(list):
    """
    A wrapper class that inherits from `list` for creating and deleting a list
    of hooks.
    """

    def __init__(self, modules: List[nn.Module], fn: Callable):
        """Creates an instance `Hooks` for creating and deleting hooks for each
        module in modules.

        Args:
            modules (nn.Module): the list of modules.
            fn (Callable): the hook function.
        """
        super().__init__([Hook(module, fn) for module in modules])

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()

    def __delitem__(self, idx):
        self[idx].remove()
        super().__delitem__(idx)

    def remove(self):
        for hook in self:
            hook.remove()
