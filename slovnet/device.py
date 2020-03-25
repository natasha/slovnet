
import torch

from .const import CUDA0, CPU


def get_device():
    if torch.cuda.is_available():
        return CUDA0
    return CPU
