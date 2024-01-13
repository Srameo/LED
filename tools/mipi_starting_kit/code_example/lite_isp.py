import torch
from torchinterp1d import Interp1d

def apply_gains(bayer_images, wbs):
    """Applies white balance to a batch of Bayer images."""
    N, C, _, _ = bayer_images.shape
    outs = bayer_images * wbs.view(N, C, 1, 1)
    return outs

def apply_ccms(images, ccms):
    """Applies color correction matrices."""
    images = images.permute(
        0, 2, 3, 1)  # Permute the image tensor to BxHxWxC format from BxCxHxW format
    images = images[:, :, :, None, :]
    ccms = ccms[:, None, None, :, :]
    outs = torch.sum(images * ccms, dim=-1)
    # Re-Permute the tensor back to BxCxHxW format
    outs = outs.permute(0, 3, 1, 2)
    return outs


def gamma_compression(images, gamma=2.2):
    """Converts from linear to gamma space."""
    outs = torch.clamp(images, min=1e-8) ** (1 / gamma)
    # outs = (1 + gamma[0]) * np.power(images, 1.0/gamma[1]) - gamma[0] + gamma[2]*images
    outs = torch.clamp((outs*255).int(), min=0, max=255).float() / 255
    return outs


def binning(bayer_images):
    """RGBG -> RGB"""
    lin_rgb = torch.stack([
        bayer_images[:,0,...],
        torch.mean(bayer_images[:, [1,3], ...], dim=1),
        bayer_images[:,2,...]], dim=1)

    return lin_rgb


def camera_response_function(images, CRF):
    E, fs = CRF # unpack CRF data

    outs = torch.zeros_like(images)
    device = images.device

    for i in range(images.shape[0]):
        img = images[i].view(3, -1)
        out = Interp1d()(E.to(device), fs.to(device), img)
        outs[i, ...] = out.view(3, images.shape[2], images.shape[3])

    outs = torch.clamp((outs*255).int(), min=0, max=255).float() / 255
    return outs


def process(bayer_images, wbs, cam2rgbs, gamma=2.2):
    """Processes a batch of Bayer RGBG images into sRGB images."""
    orig_img = bayer_images
    # White balance.
    bayer_images = apply_gains(orig_img, wbs)
    # Binning
    bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
    images = binning(bayer_images)
    # Color correction.
    images = apply_ccms(images, cam2rgbs)
    # Gamma compression.
    images = torch.clamp(images, min=0.0, max=1.0)
    images = gamma_compression(images, gamma)

    return images
