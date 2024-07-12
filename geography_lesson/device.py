"""Module to quickly get available device."""

import torch


def get_device() -> torch.device:
    """Get best available device.

    :return: best device
    :rtype: torch.device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
