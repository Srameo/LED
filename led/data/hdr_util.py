import math
import torch
from torch import nn
from torch.nn.functional import interpolate, pad, conv2d

class ClipBase(nn.Module):
    def __init__(self, clip=False) -> None:
        super().__init__()
        self._clip_func = torch.clamp if clip else diff_clamp

    def do_clip(self):
        self._clip_func = torch.clamp

    def dont_clip(self):
        self._clip_func = diff_clamp

    def forward(self, x):
        return x

def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def normalize_kernel2d(input):
    norm = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm[..., None, None])


def filter2d(input, kernel, border_type: str = 'reflect'):
    # prepare kernel
    c = input.shape[-3]
    shape = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape = _compute_padding([height, width])
    if input.dim() == 5:
        padding_shape += [0, 0]
    input = pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    out = output.view(*shape)
    return out

def get_laplacian_kernel2d(kernel_size, *, device = None, dtype = torch.float32):
    ky, kx = kernel_size
    kernel = torch.ones((ky, kx), device=device, dtype=dtype)
    mid_x = kx // 2
    mid_y = ky // 2
    kernel[mid_y, mid_x] = 1 - kernel.sum()
    return kernel

def laplacian(input, kernel_size, border_type: str = 'reflect', normalized: bool = True):
    kernel = get_laplacian_kernel2d(kernel_size, device=input.device, dtype=input.dtype)[None, ...]

    if normalized:
        kernel = normalize_kernel2d(kernel)

    return filter2d(input, kernel, border_type)


def rgb_to_grayscale(rgb):
    r, g, b = rgb.unbind(dim=-3)
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(rgb.dtype)
    l_img = l_img.unsqueeze(dim=-3)
    return l_img

def get_pyramid_gaussian_kernel():
    return (
        torch.tensor(
            [
                [
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [6.0, 24.0, 36.0, 24.0, 6.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                ]
            ]
        )
        / 256.0
    )

def pyrdown(input, border_type: str = 'reflect', align_corners: bool = False, factor: float = 2.0):
    kernel = get_pyramid_gaussian_kernel()
    channel, height, width = input.shape[-3:]
    # blur image
    x_blur = filter2d(input, kernel, border_type)

    shape = [int(float(height) / factor), int(float(width) // factor)]
    mode = 'bilinear'
    if input.dim() == 5:
        mode = 'trilinear'
        shape = [channel] + shape

    out = interpolate(
        x_blur,
        size=shape,
        mode=mode,
        align_corners=align_corners,
    )
    return out

def pyrup(input, shape, border_type: str = 'reflect', align_corners: bool = False):
    kernel = get_pyramid_gaussian_kernel()
    # upsample tensor
    mode = 'bilinear'
    if input.dim() == 5:
        mode = 'trilinear'
    else:
        shape = shape[-2:]
    x_up = interpolate(input, size=shape, mode=mode, align_corners=align_corners)

    # blurs upsampled tensor
    x_blur = filter2d(x_up, kernel, border_type)
    return x_blur

def build_pyramid(input, max_level: int, border_type: str = 'reflect', align_corners: bool = False):
    # create empty list and append the original image
    pyramid = []
    pyramid.append(input)

    # iterate and downsample
    for _ in range(max_level - 1):
        img_curr = pyramid[-1]
        img_down = pyrdown(img_curr, border_type, align_corners)
        pyramid.append(img_down)

    return pyramid

def build_laplacian_pyramid(input, max_level: int, border_type: str = 'reflect', align_corners: bool = False):
    # create gaussian pyramid
    gaussian_pyramid = build_pyramid(input, max_level, border_type, align_corners)
    laplacian_pyramid = []

    # iterate and compute difference of adjacent layers in a gaussian pyramid
    for i in range(max_level - 1):
        img_expand = pyrup(gaussian_pyramid[i + 1], gaussian_pyramid[i].shape[-3:], border_type, align_corners)
        laplacian = gaussian_pyramid[i] - img_expand
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid

def diff_clamp(x, _min, _max, k=1e-3):
    x = torch.minimum(x - _max, (x - _max) * k) + _max
    x = torch.maximum(x - _min, (x - _min) * k) + _min
    return x

def pyramid_collapse(pyramid, depth):
    for i in range(depth, 0, -1):
        pyramid[i-1] += pyrup(pyramid[i], pyramid[i-1].shape[-3:])
    return pyramid[0]


class BlendMertens(nn.Module):
    def __init__(self, contrast_weight=1.0, saturation_weight=1.0, exposure_weight=1.0, clip=False) -> None:
        super().__init__()
        self._clip_func = torch.clamp if clip else diff_clamp
        self._contrast_weight = contrast_weight
        self._saturation_weight = saturation_weight
        self._exposure_weight = exposure_weight

    def do_clip(self):
        self._clip_func = torch.clamp

    def dont_clip(self):
        self._clip_func = diff_clamp

    def get_weight(self, x):
        # contrast
        gray_x = rgb_to_grayscale(x)
        laplacian_x = laplacian(gray_x, (5, 5))
        c_weight = torch.abs(laplacian_x)
        c_weight = c_weight ** self._contrast_weight

        # saturation
        s_weight = torch.std(x, -3, keepdim=True)
        s_weight = s_weight ** self._saturation_weight

        # exposure
        sig = 0.2
        e_weight = torch.exp(-torch.pow(x - 0.5, 2) / (2 * sig * sig))
        r_w, g_w, b_w = torch.chunk(e_weight, 3, -3)
        e_weight = r_w * g_w * b_w
        e_weight = e_weight ** self._exposure_weight

        return c_weight * s_weight * e_weight + 1e-12

    def forward(self, *data):
        result = 0
        data = torch.stack(data)
        weights = self.get_weight(data)
        weight_sum = torch.sum(weights, 0, keepdim=True)
        weights = weights / weight_sum

        pyramid_depth = min(int(math.log2(512)), int(math.log2(min(data[0].shape[-2:]))))
        # pyramids
        lps = build_laplacian_pyramid(data, pyramid_depth)
        gps = build_pyramid(weights, pyramid_depth)

        # combine pyramids with weights
        result_ps = []
        for i in range(pyramid_depth):
            r_i = torch.sum(lps[i] * gps[i], 0)
            result_ps.append(r_i)
        result = self._clip_func(pyramid_collapse(result_ps, pyramid_depth-1), 0, 1)
        return result