import torch


def standardize(X: torch.Tensor) -> torch.Tensor:
    """Standardize the given tensor to have mean of 0 and standard deviation of 1.

    Args:
        X (torch.Tensor): The input tensor

    Returns:
        torch.Tensor: the standardized tensor.
    """
    return (X - X.mean()) / X.std()


def compare(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compares two tensors, and returns the number of elements that are correct in
    the range of 0 to 1.
    """
    comparison = torch.isclose(a, b, atol=1e-4, rtol=1e-3)
    return comparison.sum() / comparison.numel()


def predict_next_token(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a probability distribution for the likelihood of each token in
    the vocabulary for the next token given the input sequence.
    """
    return torch.softmax(logits[-1], dim=-1)  # the last logits gives the evidence of each token for the next token.
