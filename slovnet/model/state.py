
import torch


def load_model(model, path):
    model.load_state_dict(torch.load(path))


def dump_model(model, path):
    torch.save(model.cpu().state_dict(), path)
