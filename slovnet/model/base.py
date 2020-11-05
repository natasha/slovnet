
from torch import nn

from .state import StateMixin
from .exec import ExecMixin


class Module(nn.Module, StateMixin, ExecMixin):
    @property
    def device(self):
        """
        Return the device object with the given parameters.

        Args:
            self: (todo): write your description
        """
        for parameter in self.parameters():
            return parameter.device
