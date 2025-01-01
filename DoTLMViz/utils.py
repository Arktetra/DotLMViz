import torch


def standardize(X: torch.Tensor) -> torch.Tensor:
    """Standardize the given tensor to have mean of 0 and standard deviation of 1.

    Args:
        X (torch.Tensor): The input tensor

    Returns:
        torch.Tensor: the standardized tensor.
    """
    return (X - X.mean()) / X.std()
