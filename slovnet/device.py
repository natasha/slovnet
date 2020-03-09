
import torch


CUDA0 = 'cuda:0'
CUDA1 = 'cuda:1'
CUDA2 = 'cuda:2'
CUDA3 = 'cuda:3'
CPU = 'cpu'


def get_device():
    if torch.cuda.is_available():
        return CUDA0
    return CPU
