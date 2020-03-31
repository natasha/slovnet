
import torch

from slovnet.const import CPU


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=CPU))


def dump_model(model, path):
    torch.save(model.state_dict(), path)


class StateMixin:
    def load(self, path):
        load_model(self, path)
        return self

    def dump(self, path):
        dump_model(self, path)
