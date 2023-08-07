import torch


def random_ccm(batch_size, device='cuda'):
    """Generates random RGB -> Camera color correction matrices."""
    # Takes a random convex combination of XYZ -> Camera CCMs.
    xyz2cams = [[[1.0234, -0.2969, -0.2266],
                [-0.5625, 1.6328, -0.0469],
                [-0.0703, 0.2188, 0.6406]],
                [[0.4913, -0.0541, -0.0202],
                [-0.613, 1.3513, 0.2906],
                [-0.1564, 0.2151, 0.7183]],
                [[0.838, -0.263, -0.0639],
                [-0.2887, 1.0725, 0.2496],
                [-0.0627, 0.1427, 0.5438]],
                [[0.6596, -0.2079, -0.0562],
                [-0.4782, 1.3016, 0.1933],
                [-0.097, 0.1581, 0.5181]]]
    num_ccms = len(xyz2cams)
    xyz2cams = torch.tensor(xyz2cams, device=device).unsqueeze(0)
    weights = torch.rand((batch_size, num_ccms, 1, 1), device=device) * 1e8 + 1e-8
    weights_sum = torch.sum(weights, dim=1)
    xyz2cam = torch.sum(xyz2cams * weights, dim=1) / weights_sum

    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]], device=device)
    rgb2cam = xyz2cam @ rgb2xyz

    # Normalizes each row.
    rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdims=True)
    return rgb2cam


def random_gain(batch_size, device='cuda'):
    """Generates random gains for brightening and white balance."""
    # RGB gain represents brightening.
    rgb_gain = 1.0 / (torch.randn((batch_size), device=device) * 0.1 + 0.8)

    # Red and blue gains represent white balance.
    red_gain = torch.rand((batch_size), device=device) * (2.4 - 1.9) + 1.9
    blue_gain = torch.rand((batch_size), device=device) * (1.9 - 1.5) + 1.5
    return rgb_gain, red_gain, blue_gain


def inverse_smoothstep(image):
    """Approximately inverts a global tone mapping curve."""
    image = image.clamp(0, 1)
    return 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)


def smoothstep(image):
    image = image.clamp(0, 1)
    return 3 * image ** 2 - 2 * image ** 3


def gamma_expansion(image):
    """Converts from gamma to linear space."""
    # Clamps to prevent numerical instability of gradients near zero.
    return torch.maximum(image, torch.tensor(1e-8, device=image.device)) ** 2.2


def gamma_compression(image, gamma=2.2):
    """Converts from linear to gamma space."""
    # Clamps to prevent numerical instability of gradients near zero.
    return torch.maximum(image, torch.tensor(1e-8, device=image.device)) ** (1.0 / gamma)


def apply_ccm(image, ccm):
    """Applies a color correction matrix."""
    B, C = image.size(0), image.size(-3)
    if ccm.dim() == 3:
        tail = [1 for _ in range(image.dim() - 2)]
        ccm = ccm.reshape(B, *tail, C, C)
    image = image.transpose(-3, -1).unsqueeze(-1)   # BCHW -> BWHC -> BWHC1
    image = ccm @ image
    image = image.squeeze(-1).transpose(-3, -1)     # BWHC1 -> BWHC -> BCHW
    return image


def apply_gains(bayer_images, rgb_gains, red_gains, blue_gains, in_type='rgbg'):
    """Applies white balance gains to a batch of Bayer images."""
    tail = [1 for _ in range(bayer_images.dim() - 1)]
    rgb_gains =  rgb_gains.reshape(-1, *tail)
    red_gains =  red_gains.reshape(-1, *tail)
    blue_gains = blue_gains.reshape(-1, *tail)
    green_gains = torch.ones_like(rgb_gains)

    if in_type == 'rggb':
        gains = torch.cat([red_gains, green_gains, green_gains, blue_gains], dim=-3) * rgb_gains
    elif in_type == 'rgbg':
        gains = torch.cat([red_gains, green_gains, blue_gains, green_gains], dim=-3) * rgb_gains
    return bayer_images * gains


def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
    """Inverts gains while safely handling saturated pixels."""
    tail = [1 for _ in range(image.dim() - 1)]
    rgb_gain =  rgb_gain.reshape(-1, *tail)
    red_gain =  red_gain.reshape(-1, *tail)
    blue_gain = blue_gain.reshape(-1, *tail)

    gains = torch.cat([
        1.0 / red_gain, torch.ones_like(rgb_gain), 1.0 / blue_gain
    ], dim=-3) / rgb_gain

    # Prevents dimming of saturated pixels by smoothly masking gains near white.
    gray = torch.mean(image, dim=-3, keepdims=True)
    inflection = 0.9
    mask = (torch.maximum(gray - inflection, torch.tensor(0.0, device=gray.device)) / (1.0 - inflection)) ** 2.0
    safe_gains = torch.maximum(mask + (1.0 - mask) * gains, gains)
    return image * safe_gains


def invert_gains(image, rgb_gain, red_gain, blue_gain):
    """Inverts gains while safely handling saturated pixels."""
    tail = [1 for _ in range(image.dim() - 1)]
    rgb_gain =  rgb_gain.reshape(-1, *tail)
    red_gain =  red_gain.reshape(-1, *tail)
    blue_gain = blue_gain.reshape(-1, *tail)

    gains = torch.cat([
        1.0 / red_gain, torch.ones_like(rgb_gain), 1.0 / blue_gain
    ], dim=-3) / rgb_gain
    return image * gains


def mosaic(image, out_type):
    """Extracts RGGB Bayer planes from an RGB image."""
    # reshape into RGGB or RGBG
    r = image[..., 0:1, ::2, ::2]
    gr = image[..., 1:2, ::2, 1::2]
    gb = image[..., 1:2, 1::2, ::2]
    b = image[..., 2:3, 1::2, 1::2]
    if out_type == 'rgbg':
        out = torch.cat([r, gr, b, gb], dim=-3)
    elif out_type == 'rggb':
        out = torch.cat([r, gr, gb, b], dim=-3)
    return out


def half_size_demosaic(bayer_images, in_type):
    r = bayer_images[..., 0:1, :, :]
    gr = bayer_images[..., 1:2, :, :]
    if in_type == 'rggb':
        gb = bayer_images[..., 2:3, :, :]
        b = bayer_images[..., 3:4, :, :]
    elif in_type == 'rgbg':
        b = bayer_images[..., 2:3, :, :]
        gb = bayer_images[..., 3:4, :, :]
    g = (gr + gb) / 2
    linear_rgb = torch.cat([r, g, b], dim=-3)
    return linear_rgb


def demosaic(bayer_images, in_type='rgbg'):
    """Bilinearly demosaics a batch of RGGB Bayer images."""
    def bilinear_interpolate(x, shape):
        return torch.nn.functional.interpolate(x, shape, mode='bilinear')

    def space_to_depth(x, downscale_factor):
        return torch.nn.functional.pixel_unshuffle(x, downscale_factor)

    def depth_to_space(x, upscale_factor):
        return torch.nn.functional.pixel_shuffle(x, upscale_factor)

    # This implementation exploits how edges are aligned when upsampling with
    # torch.nn.functional.interpolate.

    if bayer_images.dim() == 5:
        B, T, C, H, W = bayer_images.shape
        bayer_images = torch.reshape(bayer_images, (B * T, C, H, W))
    elif bayer_images.dim() == 4:
        B, C, H, W = bayer_images.shape
        T = None
    assert C == 4
    shape = [H * 2, W * 2]

    red = bayer_images[:, 0:1]
    green_red = bayer_images[:, 1:2]
    if in_type == 'rggb':
        green_blue = bayer_images[:, 2:3]
        blue = bayer_images[:, 3:4]
    else:
        blue = bayer_images[:, 2:3]
        green_blue = bayer_images[:, 3:4]

    red = bilinear_interpolate(red, shape)

    green_red = torch.fliplr(green_red)
    green_red = bilinear_interpolate(green_red, shape)
    green_red = torch.fliplr(green_red)
    green_red = space_to_depth(green_red, 2)

    green_blue = torch.flipud(green_blue)
    green_blue = bilinear_interpolate(green_blue, shape)
    green_blue = torch.flipud(green_blue)
    green_blue = space_to_depth(green_blue, 2)

    green_at_red = (green_red[:, 0] + green_blue[:, 0]) / 2
    green_at_green_red = green_red[:, 1]
    green_at_green_blue = green_blue[:, 2]
    green_at_blue = (green_red[:, 3] + green_blue[:, 3]) / 2
    green_planes = [
        green_at_red, green_at_green_red, green_at_green_blue, green_at_blue
    ]
    green = depth_to_space(torch.stack(green_planes, dim=1), 2)

    blue = torch.flipud(torch.fliplr(blue))
    blue = bilinear_interpolate(blue, shape)
    blue = torch.flipud(torch.fliplr(blue))

    rgb_images = torch.cat([red, green, blue], dim=1)
    if T is not None:
        rgb_images = rgb_images.reshape(B, T, 3, H * 2, W * 2)
    return rgb_images


def reverse_isp(srgb, ccm, gains, scale, out_type='rgbg', safe_invert_gain=False):
    """reverse_isp for reverse standard rgb into RAW rgb.

    Args:
        srgb (Tensor):  the input image batch (B x C x H x W)
        ccm (Tensor):   the color correction matrix (B x C x C) or (C x C)
        gain (Tensor):  white balance (B x C)
        scale (int):    the max value of the image
        gamma (bool):   gamma correcton
        out_type (str): output data type, rgbg or rggb. Defaults to 'rgbg'.
    """
    C = srgb.size(-3)
    assert C == 3
    assert out_type in ['rgbg', 'rggb']
    assert ccm.dim() == 2 or ccm.dim() == 3

    # Approximately inverts global tone mapping.
    srgb = inverse_smoothstep(srgb)
    # Inverts gamma compression.
    linear_rgb = gamma_expansion(srgb)
    # Inverts color correction.
    ccm_inv = torch.linalg.inv(ccm)
    linear_rgb = apply_ccm(linear_rgb, ccm_inv)
    # Approximately inverts white balance and brightening.
    rgb_gain, red_gain, blue_gain = gains
    if safe_invert_gain:
        linear_rgb = safe_invert_gains(linear_rgb, rgb_gain, red_gain, blue_gain)
    else:
        linear_rgb = invert_gains(linear_rgb, rgb_gain, red_gain, blue_gain)
    # Clips saturated pixels.
    linear_rgb = torch.clamp(linear_rgb, 0.0, 1.0)
    # Applies a Bayer mosaic.
    out = mosaic(linear_rgb, out_type)

    return out * scale


def isp(raw_rgb, ccm, gains, scale, in_type='rgbg', half_size=False):
    """isp for process raw data

    Args:
        raw_rgb (_type_): _description_
        ccm (_type_): _description_
        gain (_type_): _description_
        scale (_type_): _description_
        gamma (_type_): _description_
        in_type (str, optional): _description_. Defaults to 'rgbg'.
        half_size (bool, optional): _description_. Defaults to True.
    """
    C = raw_rgb.size(-3)
    assert C == 4
    assert in_type in ['rgbg', 'rggb']
    assert ccm.dim() == 2 or ccm.dim() == 3

    # White balance.
    rgb_gain, red_gain, blue_gain = gains
    raw_rgb = apply_gains(raw_rgb, rgb_gain, red_gain, blue_gain, in_type)
    # Demosaic
    linear_rgb = half_size_demosaic(raw_rgb, in_type) if half_size else demosaic(raw_rgb, in_type)
    linear_rgb = linear_rgb.clamp(0, 1)
    C = 3
    # Color correction.
    linear_rgb = apply_ccm(linear_rgb, ccm)
    linear_rgb = linear_rgb.clamp(0, 1)
    # apply gamma correction
    srgb = gamma_compression(linear_rgb)
    # smoothstep
    srgb = smoothstep(srgb)

    return srgb * scale
