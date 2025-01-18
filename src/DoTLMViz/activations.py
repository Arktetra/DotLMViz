import torch


def gelu_new(input: torch.Tensor):
    """
    Implementation of GeLU as used by GPT2.
    """
    return (
        0.5
        * input
        * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (input + 0.044715 * torch.pow(input, 3.0))))
    )
