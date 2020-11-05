
import torch

from slovnet.const import CPU


def load_model(model, path):
    """
    Load a model from disk.

    Args:
        model: (todo): write your description
        path: (str): write your description
    """
    model.load_state_dict(torch.load(path, map_location=CPU))


def dump_model(model, path):
    """
    Dumps model to disk.

    Args:
        model: (todo): write your description
        path: (str): write your description
    """
    torch.save(model.state_dict(), path)


class StateMixin:
    def load(self, path):
        """
        Load a model.

        Args:
            self: (todo): write your description
            path: (str): write your description
        """
        load_model(self, path)
        return self

    def dump(self, path):
        """
        Dump the model to a file.

        Args:
            self: (todo): write your description
            path: (str): write your description
        """
        dump_model(self, path)
