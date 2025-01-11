import torch
from torch import nn


def rbf_kernel(X, Y, num_kernels=5, kernel_weights=None, bandwidth=None, kernel_step_multiplier=2.0):
    """
    :param X: n x d (source data)
    :param Y: m x d (target data)
    :param num_kernels: number of kernels for MK-MDD
    :param kernel_weights: kernel weights (if None then all equal to 1/num_kernels)
    :param bandwidth: rbf bandwidth parameter (if None then median heuristic used on the fly)
    :param kernel_step_multiplier:

    :return: radial basis function kernel matrices for X and Y
    """

    if bandwidth is None:
        bandwidth = calc_rbf_bandwidth(X)
        bandwidth = torch.tensor(
            [bandwidth / (kernel_step_multiplier ** ((1 / 2) * (i - num_kernels // 2))) for i in range(num_kernels)]
        )
    if kernel_weights is None:
        kernel_weights = torch.ones(num_kernels) / num_kernels

    XX = torch.sum(
        kernel_weights.unsqueeze(-1).unsqueeze(-1)
        * torch.exp(
            -torch.sum((X.unsqueeze(1).expand(-1, X.shape[0], -1)
                        - X.unsqueeze(0).expand(X.shape[0], -1, -1)) ** 2, -1)
            / (2 * bandwidth.unsqueeze(-1).unsqueeze(-1) ** 2)), dim=0)

    XY = torch.sum(
        kernel_weights.unsqueeze(-1).unsqueeze(-1)
        * torch.exp(
            -torch.sum((X.unsqueeze(1).expand(-1, Y.shape[0], -1)
                        - Y.unsqueeze(0).expand(X.shape[0], -1, -1)) ** 2, -1)
            / (2 * bandwidth.unsqueeze(-1).unsqueeze(-1) ** 2)), dim=0)

    YY = torch.sum(
        kernel_weights.unsqueeze(-1).unsqueeze(-1)
        * torch.exp(
            -torch.sum((Y.unsqueeze(1).expand(-1, Y.shape[0], -1)
                        - Y.unsqueeze(0).expand(Y.shape[0], -1, -1)) ** 2, -1)
            / (2 * bandwidth.unsqueeze(-1).unsqueeze(-1) ** 2)), dim=0)

    return XX, XY, YY


def calc_rbf_bandwidth(X):
    bandwidth = torch.median(torch.cdist(X, X, p=2))
    return bandwidth


class MKMMDLoss(nn.Module):
    def __init__(self, num_kernels=5):
        """
        :param num_kernels:
        """
        super(MKMMDLoss, self).__init__()
        self.num_kernels = num_kernels

    def forward(self, source, target, bandwidth=None, kernel_weights=None, kernel_step_multiplier=2.0):
        """
        :param source: (n_s x d) source matrix
        :param target: (n_t x d) source matrix
        :param bandwidth:
        :param kernel_weights:
        :param kernel_step_multiplier:

        :return: MK-MMD loss for source and target using radial basis function kernel as feature map
        """
        s_s, s_t, t_t = rbf_kernel(source,
                                   target,
                                   num_kernels=self.num_kernels,
                                   kernel_weights=kernel_weights,
                                   bandwidth=bandwidth,
                                   kernel_step_multiplier=kernel_step_multiplier)
        ret = 0.0
        temp1 = 0.0
        cnt1 = 0
        for i in range(len(s_s) - 1):
            temp1 += s_s[i, i + 1:].sum()
            cnt1 += len(s_s) - i - 1
        if cnt1 != 0:
            ret += temp1 / cnt1

        temp2 = 0.0
        cnt2 = 0
        for i in range(len(t_t) - 1):
            temp2 += t_t[i, i + 1:].sum()
            cnt2 += len(t_t) - i - 1
        if cnt2 != 0:
            ret += temp2 / cnt2

        ret -= 2*s_t.mean()
        return ret
