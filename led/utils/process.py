"""Forward processing of raw data to sRGB images.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

import numpy as np
import torch
import contextlib

from os.path import join

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Interp1d(torch.autograd.Function):
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.

        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.

        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        torch.searchsorted(v['x'].contiguous(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)

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


def gamma_compression_grad(images, gamma=2.2):
    """Converts from linear to gamma space."""
    outs = torch.clamp(images, min=1e-8) ** (1 / gamma)
    # outs = (1 + gamma[0]) * np.power(images, 1.0/gamma[1]) - gamma[0] + gamma[2]*images
    return outs


def binning(bayer_images):
    """RGBG -> RGB"""
    lin_rgb = torch.stack([
        bayer_images[:,0,...],
        torch.mean(bayer_images[:, [1,3], ...], dim=1),
        bayer_images[:,2,...]], dim=1)

    return lin_rgb


def process(bayer_images, wbs, cam2rgbs, gamma=2.2, CRF=None):
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
    if CRF is None:
        images = gamma_compression(images, gamma)
    else:
        images = camera_response_function(images, CRF)

    return images


def process_grad(bayer_images, wbs, cam2rgbs, gamma=2.2, CRF=None):
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
    if CRF is None:
        images = gamma_compression_grad(images, gamma)
    else:
        images = camera_response_function_grad(images, CRF)

    return images


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

def camera_response_function_grad(images, CRF):
    E, fs = CRF # unpack CRF data

    outs = torch.zeros_like(images)
    device = images.device

    for i in range(images.shape[0]):
        img = images[i].view(3, -1)
        out = Interp1d()(E.to(device), fs.to(device), img)
        outs[i, ...] = out.view(3, images.shape[2], images.shape[3])

    return outs

def raw2rgb(packed_raw, raw, CRF=None, gamma=2.2):
    """Raw2RGB pipeline (preprocess version)"""
    wb = np.array(raw.camera_whitebalance)
    wb /= wb[1]
    cam2rgb = raw.rgb_camera_matrix[:3, :3]

    if isinstance(packed_raw, np.ndarray):
        packed_raw = torch.from_numpy(packed_raw).float()

    wb = torch.from_numpy(wb).float().to(packed_raw.device)
    cam2rgb = torch.from_numpy(cam2rgb).float().to(packed_raw.device)

    out = process(packed_raw[None], wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=gamma, CRF=CRF)[0, ...].numpy()

    return out


def raw2rgb_v2(packed_raw, wb, ccm, CRF=None, gamma=2.2): # RGBG
    packed_raw = torch.from_numpy(packed_raw).float()
    wb = torch.from_numpy(wb).float()
    cam2rgb = torch.from_numpy(ccm).float()
    out = process(packed_raw[None], wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=gamma, CRF=CRF)[0, ...].numpy()
    return out


def raw2rgb_torch(packed_raw, wb, ccm, CRF=None, gamma=2.2, batch=False): # RGBG
    if batch:
        out = process(packed_raw, wbs=wb, cam2rgbs=ccm, gamma=gamma, CRF=CRF)
    else:
        out = process(packed_raw[None], wbs=wb[None], cam2rgbs=ccm[None], gamma=gamma, CRF=CRF)
    return out

def raw2rgb_torch_grad(packed_raw, wb, ccm, CRF=None, gamma=2.2): # RGBG
    out = process_grad(packed_raw, wbs=wb, cam2rgbs=ccm, gamma=gamma, CRF=CRF)
    return out

def raw2rgb_postprocess(packed_raw, raw, CRF=None):
    """Raw2RGB pipeline (postprocess version)"""
    assert packed_raw.ndimension() == 4 and packed_raw.shape[0] == 1
    wb = np.array(raw.camera_whitebalance)
    wb /= wb[1]
    cam2rgb = raw.rgb_camera_matrix[:3, :3]

    wb = torch.from_numpy(wb[None]).float().to(packed_raw.device)
    cam2rgb = torch.from_numpy(cam2rgb[None]).float().to(packed_raw.device)
    out = process(packed_raw, wbs=wb, cam2rgbs=cam2rgb, gamma=2.2, CRF=CRF)
    return out

def read_wb_ccm(raw):
    wb = np.array(raw.camera_whitebalance)
    wb /= wb[1]
    wb = wb.astype(np.float32)
    ccm = raw.rgb_camera_matrix[:3, :3].astype(np.float32)
    return wb, ccm


def read_emor(address):
    def _read_curve(lst):
        curve = [l.strip() for l in lst]
        curve = ' '.join(curve)
        curve = np.array(curve.split()).astype(np.float32)
        return curve

    with open(address) as f:
        lines = f.readlines()
        k = 1
        E = _read_curve(lines[k:k+256])
        k += 257
        f0 = _read_curve(lines[k:k+256])
        hs = []
        for _ in range(25):
            k += 257
            hs.append(_read_curve(lines[k:k+256]))

        hs = np.array(hs)

        return E, f0, hs


def read_dorf(address):
    with open(address) as f:
        lines = f.readlines()
        curve_names = lines[0::6]
        Es = lines[3::6]
        Bs = lines[5::6]

        Es = [np.array(E.strip().split()).astype(np.float32) for E in Es]
        Bs = [np.array(B.strip().split()).astype(np.float32) for B in Bs]

    return curve_names, Es, Bs


def load_CRF(EMoR_path):
    # init CRF function
    fs = np.loadtxt(join(EMoR_path, 'CRF_SonyA7S2_5.txt'))
    E, _, _ = read_emor(join(EMoR_path, 'emor.txt'))
    E = torch.from_numpy(E).repeat(3, 1)
    fs = torch.from_numpy(fs)
    CRF = (E, fs)
    return CRF
