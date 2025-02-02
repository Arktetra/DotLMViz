import torch

from typing import Callable, List

from .utils import convert_to_points


def epanechnikov(v: torch.Tensor, h: float):
    """
    The Epanechnikov kernel function.
    """
    return torch.where(torch.abs(v / h) <= 1, 0.75 * (1 - (v / h) ** 2), 0)


class KernelDensityEstimator:
    """
    A kernel density estimator that will be created according to the kernel
    function and bandwidth passed.
    """

    def __init__(self, kernel: Callable = epanechnikov, bandwidth: float = 0.17):
        self.kernel = kernel
        self.bandwidth = bandwidth

    def estimate(self, data: torch.Tensor, start: float, end: float, steps: int = None) -> List[List[float]]:
        """Estimates the probability density function of the data, and returns
        a list of points consisting of the probability density for the provided
        range.

        Args:
            data (torch.Tensor): the data of which the probability density is to be estimated.
            start (float): the starting value of the x values.
            end (float): the ending value of the x values.
            steps (int, optional): number of points between start and end. Defaults to None.

        Returns:
            List[List[float]]: A list of points consisting of the probability density
            for the provided range.
        """

        steps = 1 if steps is None else steps
        x = torch.linspace(start, end, steps).reshape(-1, 1, 1)
        y = self.kernel(x - data, self.bandwidth).mean(dim=[1, 2])
        return convert_to_points(x.squeeze(), y)
