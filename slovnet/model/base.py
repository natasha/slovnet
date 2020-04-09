
from torch import nn

from .state import StateMixin


class Module(nn.Module, StateMixin):
    @property
    def device(self):
        for parameter in self.parameters():
            return parameter.device
