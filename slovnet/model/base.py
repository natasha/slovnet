
from torch import nn

from .state import StateMixin
from .exec import ExecMixin


class Module(nn.Module, StateMixin, ExecMixin):
    @property
    def device(self):
        for parameter in self.parameters():
            return parameter.device
