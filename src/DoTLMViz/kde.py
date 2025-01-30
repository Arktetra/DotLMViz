import torch

from typing import Callable

from .utils import convert_to_points


def epanechnikov(v: torch.Tensor, h: float):
    """
    The Epanechnikov kernel function.
    """
    return torch.where(torch.abs(v / h) <= 1, 0.75 * (1 - (v / h) ** 2), 0)


class KernelDensityEstimator:
    def __init__(self, kernel: Callable = epanechnikov, bandwidth: float = 0.17):
        self.kernel = kernel
        self.bandwidth = bandwidth

    def estimate(self, data: torch.Tensor, start: float, end: float, steps: int = None):
        steps = 1 if steps is None else steps
        x = torch.linspace(start, end, steps).reshape(-1, 1, 1)
        y = self.kernel(x - data, self.bandwidth).mean(dim=[1, 2])
        return convert_to_points(x.squeeze(), y)
