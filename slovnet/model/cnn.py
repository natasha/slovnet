
from torch import nn

from .base import Module


class CNNEncoderLayer(Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(CNNEncoderLayer, self).__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_dim, out_dim, kernel_size,
            padding=padding
        )
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_dim)

    def __call__(self, input):
        x = self.conv(input)
        x = self.relu(x)
        return self.norm(x)


def gen_cnn_encoder_layers(input_dim, layer_dims, kernel_size):
    dims = [input_dim] + layer_dims
    for index in range(1, len(dims)):
        in_dim = dims[index - 1]
        out_dim = dims[index]
        yield CNNEncoderLayer(in_dim, out_dim, kernel_size)


class CNNEncoder(Module):
    def __init__(self, input_dim, layer_dims, kernel_size):
        super(CNNEncoder, self).__init__()

        layers = gen_cnn_encoder_layers(input_dim, layer_dims, kernel_size)
        self.layers = nn.ModuleList(layers)
        self.dim = layer_dims[-1]

    def forward(self, input, mask=None):  # batch x seq x emb
        input = input.transpose(2, 1)  # batch x emb x seq

        if mask is not None:
            mask = mask.unsqueeze(1)  # batch x 1 x seq

        for layer in self.layers:
            input = layer(input)  # batch x dim x seq

            if mask is not None:
                input[mask.expand_as(input)] = 0

        return input.transpose(2, 1)  # batch x seq x dim
