
from torch import nn

from .base import Module


class CNNEncoderLayer(Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        """
        Initialize a convolution layer.

        Args:
            self: (todo): write your description
            in_dim: (int): write your description
            out_dim: (int): write your description
            kernel_size: (int): write your description
        """
        super(CNNEncoderLayer, self).__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_dim, out_dim, kernel_size,
            padding=padding
        )
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_dim)

    def __call__(self, input):
        """
        Implement operator.

        Args:
            self: (todo): write your description
            input: (array): write your description
        """
        x = self.conv(input)
        x = self.relu(x)
        return self.norm(x)


def gen_cnn_encoder_layers(input_dim, layer_dims, kernel_size):
    """
    Generate encoder_encoder.

    Args:
        input_dim: (str): write your description
        layer_dims: (todo): write your description
        kernel_size: (int): write your description
    """
    dims = [input_dim] + layer_dims
    for index in range(1, len(dims)):
        in_dim = dims[index - 1]
        out_dim = dims[index]
        yield CNNEncoderLayer(in_dim, out_dim, kernel_size)


class CNNEncoder(Module):
    def __init__(self, input_dim, layer_dims, kernel_size):
        """
        Initialize the kernel.

        Args:
            self: (todo): write your description
            input_dim: (int): write your description
            layer_dims: (int): write your description
            kernel_size: (int): write your description
        """
        super(CNNEncoder, self).__init__()

        layers = gen_cnn_encoder_layers(input_dim, layer_dims, kernel_size)
        self.layers = nn.ModuleList(layers)
        self.dim = layer_dims[-1]

    def forward(self, input, mask=None):  # batch x seq x emb
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            mask: (todo): write your description
        """
        input = input.transpose(2, 1)  # batch x emb x seq

        if mask is not None:
            mask = mask.unsqueeze(1)  # batch x 1 x seq

        for layer in self.layers:
            input = layer(input)  # batch x dim x seq

            if mask is not None:
                input[mask.expand_as(input)] = 0

        return input.transpose(2, 1)  # batch x seq x dim
