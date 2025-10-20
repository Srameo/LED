from abc import ABC
import math
import torch
import numpy as np
from scipy import stats

def _unpack_bayer(x):
    _, h, w = x.shape
    H = h*2
    W = w*2
    out = np.zeros((H, W))

    out[0:H:2, 0:W:2] = x[0]
    out[0:H:2, 1:W:2] = x[1]
    out[1:H:2, 1:W:2] = x[2]
    out[1:H:2, 0:W:2] = x[3]
    return out

def _pack_bayer(raw):
    h, w = raw.shape
    out = np.concatenate((raw[np.newaxis, 0:h:2, 0:w:2],
                            raw[np.newaxis, 0:h:2, 1:w:2],
                            raw[np.newaxis, 1:h:2, 1:w:2],
                            raw[np.newaxis, 1:h:2, 0:w:2]), axis=0)
    return out

def _unpack_bayer_torch(x):
    _, h, w = x.shape
    H = h*2
    W = w*2
    out = torch.zeros((H, W))

    out[0:H:2, 0:W:2] = x[0]
    out[0:H:2, 1:W:2] = x[1]
    out[1:H:2, 1:W:2] = x[2]
    out[1:H:2, 0:W:2] = x[3]
    return out

# def _pack_bayer_torch(raw):
#     h, w = raw.shape
#     out = torch.cat((raw[None, 0:h:2, 0:w:2], # R
#                      raw[None, 0:h:2, 1:w:2], # G1
#                      raw[None, 1:h:2, 1:w:2], # B
#                      raw[None, 1:h:2, 0:w:2]  # G2
#                     ), dim=0)
#     return out

def _pack_bayer_torch(raw):
    h, w = raw.size(-2), raw.size(-1)
    out = torch.cat((raw[..., None, 0:h:2, 0:w:2], # R
                     raw[..., None, 0:h:2, 1:w:2], # G1
                     raw[..., None, 1:h:2, 1:w:2], # B
                     raw[..., None, 1:h:2, 0:w:2]  # G2
                    ), dim=-3)
    return out

def _pack_batch_bayer_torch(raw):
    _, h, w = raw.shape
    out = torch.cat((raw[:, None, 0:h:2, 0:w:2], # R
                     raw[:, None, 0:h:2, 1:w:2], # G1
                     raw[:, None, 1:h:2, 1:w:2], # B
                     raw[:, None, 1:h:2, 0:w:2]  # G2
                    ), dim=1)
    return out

def torch_shot_noise(x, k):
    return torch.poisson(x / k) * k - x

def torch_gaussian_noise(x, scale, loc=0):
    return torch.randn_like(x) * scale + loc
    # return torch.zeros_like(x).normal_(loc, scale)

def torch_turkey_lambda_noise(x, scale, t_lambda=1.4):
    def turkey_lambda_ppf(p, t_lambda):
        # assert not torch.any(torch.tensor(t_lambda == 0.0))
        return 1 / t_lambda * (p ** t_lambda - (1 - p) ** t_lambda)

    epsilon = 1e-10
    U = torch.rand_like(x) * (1 - 2 * epsilon) + epsilon
    Y = turkey_lambda_ppf(U, t_lambda + 1e-8) * scale

    return Y

def torch_quant_noise(x, q):
    return (torch.rand_like(x) - 0.5) * q

# def torch_row_noise(x, scale, loc=0):
#     _, H, W = x.shape
#     noise = torch.zeros((H * 2, 1), device=x.device).normal_(loc, scale).repeat((1, W * 2))
#     return _pack_bayer_torch(noise)

def torch_row_noise(x, scale, loc=0):
    if x.dim() == 4:
        B, _, H, W = x.shape
        noise = (torch.randn((B, H * 2, 1), device=x.device) * scale + loc).repeat((1, 1, W * 2))
    elif x.dim() == 5:
        B, T, _, H, W = x.shape
        noise = (torch.randn((B, T, H * 2, 1), device=x.device) * scale + loc).repeat((1, 1, 1, W * 2))
    elif x.dim() == 3:
        _, H, W = x.shape
        noise = torch.zeros((H * 2, 1), device=x.device).normal_(loc, scale).repeat((1, W * 2))
    else:
        raise NotImplementedError()
    return _pack_bayer_torch(noise)

def torch_batch_row_noise(x, scale, loc=0):
    B, _, H, W = x.shape
    noise = (torch.randn((B, H * 2, 1), device=x.device) * scale + loc).repeat((1, 1, W * 2))
    return _pack_batch_bayer_torch(noise)

def numpy_shot_noise(x, k):
    # print(x, k)
    return np.random.poisson(x / k).astype(np.float32) * k - x

def numpy_gaussian_noise(x, scale):
    return stats.norm.rvs(scale=scale, size=x.shape).astype(np.float32)

def numpy_turkey_lambda_noise(x, scale, t_lambda=1.4):
    return stats.tukeylambda.rvs(t_lambda, scale=scale, size=[*x.shape]).astype(np.float32)

def numpy_row_noise(x, scale):
    _, H, W = x.shape
    noise = np.random.randn(H * 2, 1).astype(np.float32) * scale
    noise = np.repeat(noise, W * 2, 1)
    return _pack_bayer(noise)

def numpy_quant_noise(x, q):
    return np.random.uniform(low=-0.5*q, high=0.5*q, size=x.shape)

class Engine(ABC):
    @staticmethod
    def uniform(min, max, shape=None):
        pass

    @staticmethod
    def randint(min, max, shape=None):
        pass

    @staticmethod
    def randn(shape=None):
        pass

    @staticmethod
    def log(x):
        pass

    @staticmethod
    def exp(x):
        pass

    @staticmethod
    def to_engine_type(x):
        pass

    @staticmethod
    def shot_noise(x, k):
        pass

    @staticmethod
    def gaussian_noise(x, scale):
        pass

    @staticmethod
    def turkey_lambda_noise(x, scale, t_lambda):
        pass

    @staticmethod
    def row_noise(x, scale):
        pass

    @staticmethod
    def quant_noise(x, q):
        pass

class NumpyEngine(Engine):
    @staticmethod
    def uniform(min, max, shape=None):
        if shape == None:
            return np.random.uniform(min, max)
        return np.random.uniform(min, max, size=shape)

    @staticmethod
    def randint(min, max, shape=None):
        if shape == None:
            return np.random.randint(min, max)
        return np.random.randint(min, max, size=shape)

    @staticmethod
    def randn(shape=None):
        if shape == None:
            return np.random.randn()
        return np.random.randn(shape)

    @staticmethod
    def log(x):
        return np.log(x)

    @staticmethod
    def exp(x):
        return np.exp(x)

    @staticmethod
    def to_engine_type(x):
        return np.array(x)

    @staticmethod
    def shot_noise(x, k):
        return numpy_shot_noise(x, k)

    @staticmethod
    def gaussian_noise(x, scale):
        return numpy_gaussian_noise(x, scale)

    @staticmethod
    def turkey_lambda_noise(x, scale, t_lambda):
        return numpy_turkey_lambda_noise(x, scale, t_lambda)

    @staticmethod
    def row_noise(x, scale):
        return numpy_row_noise(x, scale)

    @staticmethod
    def quant_noise(x, q):
        return numpy_quant_noise(x, q)

class TorchEngine(Engine):
    @staticmethod
    def uniform(min, max, shape=1, device='cpu'):
        if shape != 1:
            return torch.rand((shape,), device=device) * (max - min) + min
        return torch.rand((shape,)).item() * (max - min) + min

    @staticmethod
    def randint(min, max, shape=1, device='cpu'):
        if shape != 1:
            return torch.randint(min, max, (shape,), device=device)
        return torch.randint(min, max, (shape,)).item()

    @staticmethod
    def randn(shape=1, device='cpu'):
        if shape != 1:
            return torch.randn((shape,), device=device)
        return torch.randn((shape,)).item()

    @staticmethod
    def log(x):
        return math.log(x)

    @staticmethod
    def exp(x):
        return math.exp(x)

    @staticmethod
    def to_engine_type(x):
        return torch.tensor(x)

    @staticmethod
    def shot_noise(x, k):
        return torch_shot_noise(x, k)

    @staticmethod
    def gaussian_noise(x, scale):
        return torch_gaussian_noise(x, scale)

    @staticmethod
    def turkey_lambda_noise(x, scale, t_lambda):
        return torch_turkey_lambda_noise(x, scale, t_lambda)

    @staticmethod
    def row_noise(x, scale):
        return torch_row_noise(x, scale)

    @staticmethod
    def quant_noise(x, q):
        return torch_quant_noise(x, q)


class TorchBatchEngine(TorchEngine):
    def __init__(self, use_hflip=True, use_rot=True) -> None:
        super().__init__()
        self.use_hflip = use_hflip
        self.use_rot = use_rot

    @staticmethod
    def uniform(min, max, shape=None, device='cpu'):
        if shape is not None:
            return torch.rand((shape,), device=device) * (max - min) + min
        return torch.rand((1,)).item() * (max - min) + min

    @staticmethod
    def turkey_lambda_noise(x, scale, t_lambda):
        return torch_turkey_lambda_noise(x, scale, t_lambda)

    @staticmethod
    def log(x):
        return torch.log(x)

    @staticmethod
    def exp(x):
        return torch.exp(x)

    @staticmethod
    def row_noise(x, scale):
        return torch_batch_row_noise(x, scale)

    def augment(self, *datas):
        hflip = self.use_hflip and self.randint(0, 2) == 1
        vflip = self.use_rot and self.randint(0, 2) == 1
        rot90 = self.use_rot and self.randint(0, 2) == 1
        if hflip:
            datas = [torch.flip(data, (3,)) for data in datas]
        if vflip:
            datas = [torch.flip(data, (2,)) for data in datas]
        if rot90:
            datas = [torch.permute(data, (0, 1, 3, 2)) for data in datas]
        return datas