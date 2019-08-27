
from collections import OrderedDict

from torch import nn


class ContextModel(nn.Module):
    pass


def CNNLayer(in_dim, out_dim, kernel_size):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv1d(
            in_dim, out_dim, kernel_size,
            padding=padding
        )),
        ('relu', nn.ReLU()),
        ('norm', nn.BatchNorm1d(out_dim))
    ]))


def CNNLayers(input_dim, layer_dims, kernel_size):
    dims = [input_dim] + layer_dims
    for index in range(1, len(dims)):
        in_dim = dims[index - 1]
        out_dim = dims[index]
        yield CNNLayer(in_dim, out_dim, kernel_size)


class CNNContextModel(ContextModel):
    def __init__(self, input_dim, layer_dims, kernel_size):
        super(CNNContextModel, self).__init__()
        self.layers = nn.Sequential(*CNNLayers(
            input_dim, layer_dims, kernel_size
        ))
        self.dim = layer_dims[-1]

    def forward(self, input):  # batch x seq x emb
        input = input.transpose(2, 1)  # batch x emb x seq
        context = self.layers(input)  # batch x dim x seq
        context = context.transpose(2, 1)  # batch x seq x dim
        return context


class RNNContextModel(ContextModel):
    def __init__(self, input_dim, hidden_dim, layers_count, RNN=nn.GRU):
        super(RNNContextModel, self).__init__()
        self.rnn = RNN(
            input_dim,
            hidden_dim,
            num_layers=layers_count,
            bidirectional=True,
            batch_first=True
        )
        self.dim = hidden_dim * 2  # forward + backward

    def forward(self, input):
        context, _ = self.rnn(input)
        return context
