
import torch


CUDA0 = 'cuda:0'
CPU = 'cpu'


def get_device():
    if torch.cuda.is_available():
        return CUDA0
    return CPU
