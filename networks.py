import torch
from torch import nn


class LinBlockFirst(nn.Module):
    def __init__(self, dim_in, dim_out, dropout_rate):
        super(LinBlockFirst, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=dim_in, out_features=dim_out),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.block(x)


class LinBlockMiddle(nn.Module):
    def __init__(self, dim_in, dim_out, dropout_rate):
        super(LinBlockMiddle, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=dim_in, out_features=dim_out),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.block(x)


# Deep Adaptation Network
class DAN(nn.Module):

    def __init__(self,
                 dim,
                 dropout_rate=0.3,
                 num_kernels=5,
                 out_dim=1,
                 num_hidden_layers=2,
                 embed_size=8,
                 num_mmd_layers=2,
                 bandwidth=1,
                 kernel_step_multiplier=2.0):
        """

        :param dim: input dimension size (number of features)
        :param dropout_rate:
        :param num_kernels:
        :param out_dim:
        :param num_hidden_layers:
        :param embed_size:
        :param num_mmd_layers:
        :param bandwidth:
        :param kernel_step_multiplier:
        """
        super(DAN, self).__init__()
        self.dim = dim
        self.dropout_rate = dropout_rate
        self.kernel_weights = nn.Parameter(torch.tensor([[1 / num_kernels] * num_kernels] * num_mmd_layers))

        # First layer
        hidden_layers = [LinBlockFirst(dim_in=dim,
                                       dim_out=embed_size * (2 ** num_hidden_layers),
                                       dropout_rate=dropout_rate)]
        # Middle layers
        for i in range(num_hidden_layers - 1, 0, -1):
            hidden_layers += [LinBlockMiddle(dim_in=embed_size * (2 ** (i + 1)),
                                             dim_out=embed_size * (2 ** i),
                                             dropout_rate=dropout_rate)]
        self.hidden_layers = nn.Sequential(*hidden_layers)

        mmd_layers = [nn.Linear(in_features=embed_size * 2, out_features=embed_size)]
        for j in range(num_mmd_layers - 1):
            mmd_layers += [nn.Linear(in_features=embed_size, out_features=embed_size)]
        self.mmd_layers = nn.ModuleList(mmd_layers)

        self.output_layer = nn.Linear(in_features=embed_size, out_features=out_dim)

    def forward(self, data):
        """

        :param data:
        :return:
        """
        embeddings = []
        data = self.hidden_layers(data)
        for layer in self.mmd_layers:
            data = layer(data)
            embeddings.append(data)
            data = nn.functional.tanh(data)
        pred = self.output_layer(data)
        return [embeddings, pred]
