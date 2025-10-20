import torch
from torch import nn
from torch.nn import functional as F

class LayerNorm2d(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, num_features, eps=1e-6, affine=True, data_format="channels_first", track_running_stats=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features)) if affine else None
        self.bias = nn.Parameter(torch.zeros(num_features)) if affine else None
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.num_features = (num_features, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.num_features, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if self.weight is not None:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ScaleNorm2d(nn.Module):
    def __init__(self, num_features, bias=True, *, init_weight=1.0, init_bias=0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1) * init_weight, requires_grad=True)
        self.bias = nn.Parameter(torch.ones(1, num_features, 1, 1) * init_bias, requires_grad=bias)

    def forward(self, x):
        return self.weight * x + self.bias

class MultipleScaleNorm2d(nn.Module):
    def __init__(self, num_features, bias=True, numbers=1, *, init_weight=1.0, init_bias=0.0) -> None:
        super().__init__()
        self.norms = nn.ModuleList()
        for _ in range(numbers):
            self.norms.append(ScaleNorm2d(
                num_features,
                bias=bias,
                init_weight=init_weight,
                init_bias=init_bias
            ))
        # self.deploy_norm = ScaleNorm2d(num_features, False)
        # self.deploy_norm.weight.requires_grad_(False)
        self.numbers = numbers

    def forward(self, x, idx=0):
        assert idx < self.numbers
        return self.norms[idx](x)
