import torch
from torch import nn
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import os

from copy import deepcopy
from ultraled.archs import build_network
from ultraled.losses import build_loss
from ultraled.metrics import calculate_metric
from ultraled.utils import get_root_logger, imwrite, tensor2img
from ultraled.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

from ultraled.utils import load_CRF, raw2rgb_torch, raw2rgb_torch_grad
from ultraled.data.hdr_util import BlendMertens
from ultraled.data.noise_util_rawhdr import NoiseGenerator

import yaml
import torch.nn.functional as F

def sum_img_and_noise(img, noises):
    for noise in noises:
        img += noise
    return img

def gamma_correct(linear, eps=None):
    if eps is None:
        eps = torch.finfo(torch.float32).eps
    srgb0 = 323 / 25 * linear
    srgb1 = (211 * torch.maximum(torch.tensor(eps), linear)**(5 / 12) - 11) / 200
    return torch.where(linear <= 0.0031308, srgb0, srgb1)


def gamma_expansion(srgb, eps=None):
    if eps is None:
        eps = torch.finfo(torch.float32).eps
    linear0 = 25 / 323 * srgb
    linear1 = torch.maximum(torch.tensor(eps), ((200 * srgb + 11) / (211)))**(12 / 5)
    return torch.where(srgb <= 0.04045, linear0, linear1)

def half_size_demosaic(bayer_images):
    r = bayer_images[..., 0:1, :, :]
    gr = bayer_images[..., 1:2, :, :]
    b = bayer_images[..., 2:3, :, :]
    gb = bayer_images[..., 3:4, :, :]
    g = (gr + gb) / 2
    linear_rgb = torch.cat([r, g, b], dim=-3)
    return linear_rgb

def apply_gains(bayer_images, wbs):
    """Applies white balance to a batch of Bayer images."""
    B, N, C, _, _ = bayer_images.shape
    outs = bayer_images * wbs.view(B, -1, C, 1, 1)
    return outs

def apply_ccms(images, ccms):
    """Applies color correction matrices."""
    images = images.permute(
        0, 1, 3, 4, 2)  # Permute the image tensor to BxHxWxC format from BxCxHxW format
    images = images[:, :, :, :, None, :]
    ccms = ccms[:, :, None, None, :, :]
    outs = torch.sum(images * ccms, dim=-1)
    # Re-Permute the tensor back to BxCxHxW format
    outs = outs.permute(0, 1, 4, 2, 3)
    return outs


def tiny_isp(im, wb, ccm):
    im = half_size_demosaic(im)
    im = apply_gains(im, wb)
    im = apply_ccms(im, ccm)
    return gamma_correct(im).clip(0, 1)

def reverse_tiny_isp(srgb, wb, ccm):
    raw = gamma_expansion(srgb)
    raw = apply_ccms(raw, torch.inverse(ccm))
    raw = apply_gains(raw, 1.0 / wb)
    # expand to 4 channels
    raw_g = raw[..., 1:2, :, :]
    raw = torch.cat([raw, raw_g], dim=-3)
    return raw

def exposure_fusion_from_raw(ims, wb, ccm, blend_menten):
    """
    ims: B, N, C, H, W
    wb: B, 4
    ccm: B, 3, 3
    """
    wb = wb[..., :3] # B, 1, 3
    ccm = ccm.unsqueeze(1) # B, 1, 3, 3
    srgbs = tiny_isp(ims, wb, ccm) # B, N, 3, H, W
    merged = blend_menten(*[srgbs[:, i] for i in range(srgbs.shape[1])]) # B, 3, H, W
    merged_raw = reverse_tiny_isp(merged.unsqueeze(1), wb, ccm).squeeze(1) # B, 4, H, W
    return merged_raw


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, 'r') as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])



def load_network(net, load_path, strict=True, param_key='params'):
    """Load network.

    Args:
        load_path (str): The path of networks to be loaded.
        net (nn.Module): Network.
        strict (bool): Whether strictly loaded.
        param_key (str): The parameter key of loaded network. If set to
            None, use the root 'path'.
            Default: 'params'.
    """
    load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
    if param_key is not None:
        if param_key not in load_net and 'params' in load_net:
            param_key = 'params'
            print('Loading: params_ema does not exist, use params.')
        load_net = load_net[param_key]
    print(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    net.load_state_dict(load_net, strict=strict)

class IlluminanceCorrect(nn.Module):
    def __init__(self):
        super(IlluminanceCorrect, self).__init__()

    # Illuminance Correction
    def forward(self, predict, source):
        if predict.shape[0] != 1:
            output = torch.zeros_like(predict)
            if source.shape[0] != 1:
                for i in range(predict.shape[0]):
                    output[i:i+1, ...] = self.correct(predict[i:i+1, ...], source[i:i+1, ...])
            else:
                for i in range(predict.shape[0]):
                    output[i:i+1, ...] = self.correct(predict[i:i+1, ...], source)
        else:
            output = self.correct(predict, source)
        return output

    def correct(self, predict, source):
        N, C, H, W = predict.shape
        predict = torch.clamp(predict, 0, 1)
        assert N == 1
        output = torch.zeros_like(predict, device=predict.device)
        pred_c = predict[source != 1]
        source_c = source[source != 1]

        num = torch.dot(pred_c, source_c)
        den = torch.dot(pred_c, pred_c)
        output = num / den * predict

        return output


@MODEL_REGISTRY.register()
class RatioMapEstimatorModel(BaseModel):

    def __init__(self, opt):
        super(RatioMapEstimatorModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        self.blend_merten = BlendMertens(contrast_weight=1.0, saturation_weight=1.0, exposure_weight=1.0, clip=True)
        self.noise_gen = NoiseGenerator(**self.opt['noise_g'])

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.opt.get('CRF_path', None) is not None:
            self.CRF = load_CRF(self.opt['CRF_path'])
        else:
            self.CRF = None

        self.correct = self.opt['val'].get('illumination_correct', False)
        if self.correct:
            self.corrector = IlluminanceCorrect()
        self.metric_in_srgb = self.opt.get('metric_in_srgb', False)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('srgb_opt'):
            self.post_process = lambda x, wb, ccm: raw2rgb_torch_grad(x, wb, ccm, self.CRF)
        else:
            self.post_process = lambda x, wb, ccm: x

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        learnable_layers = train_opt.get('learnable_layers', None)
        learnable_keys = train_opt.get('learnable_keys', None)
        optim_params = []
        if learnable_layers is None and learnable_keys is None:
            for k, v in self.net_g.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Params {k} will not be optimized.')
        else:
            if learnable_keys is not None:
                logger = get_root_logger()
                logger.info(f'Using \'learnable_keys\' for query training paarameters ...')
                for k, v in self.net_g.named_parameters():
                    for l_key in learnable_keys:
                        if l_key in k:
                            optim_params.append(v)
                            break
                assert len(optim_params) > 0
            if learnable_layers is not None:
                logger = get_root_logger()
                logger.info(f'Using \'learnable_layers\' for query training paarameters ...')
                for layer in learnable_layers:
                    if hasattr(self.net_g, layer):
                        optim_params.extend(list(eval(f'self.net_g.{layer}').parameters()))
                    else:
                        logger = get_root_logger()
                        logger.error(f'Layer {layer} is not in {self.net_g.__name__}.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):

        self.intact = data['intact'].to(self.device)
        self.lq_clean = data['lq_clean'].to(self.device)
        self.ccm = data['ccm'].to(self.device)
        self.wb = data['wb'].to(self.device)
        self.ratio = data['ratio'].to(self.device)
        self.ratio_all = data['ratio1'].to(self.device)

        mexp_lq = data['gt'].to(self.device)
        fused_raw = torch.clamp(exposure_fusion_from_raw(mexp_lq, self.wb, self.ccm, self.blend_merten), 1e-8)
        self.gt =  self.lq_clean / fused_raw
        self.ratio_all = self.ratio_all.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        self.gt = self.gt.clip(1 / self.ratio_all, self.ratio_all)

        # add noise
        lq_im_patch = torch.clamp(self.lq_clean , min=0) * (16383 - 512) / self.ratio_all
        im_patch, noise1 = self.noise_gen(lq_im_patch)
        lq_im_patch = sum_img_and_noise(im_patch, noise1) / (16383 - 512) * self.ratio_all

        self.lq = lq_im_patch


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        index = self.opt['network_g']
        net = index['type']

        if str(net) == 'UNetArch' or str(net) == 'Restormer':
            self.output = self.net_g(self.lq)
        else:
            self.output = self.net_g(self.lq, self.ratiomap)

        self.output = self.post_process(self.output, self.wb, self.ccm)
        self.gt = self.post_process(self.gt, self.wb, self.ccm)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:

            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        # padding
        h, w = self.lq.shape[2:]
        pad_h = 16 - (h % 16) if h % 16 != 0 else 0
        pad_w = 16 - (w % 16) if w % 16 != 0 else 0
        
        self.gt = self.gt.squeeze(0)
        self.lq = nn.functional.pad(self.lq, [0, pad_w, 0, pad_h], mode='replicate')
        self.gt = nn.functional.pad(self.gt, [0, pad_w, 0, pad_h], mode='replicate')

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
                # illumination correction
                if self.correct:
                    self.output = self.corrector(self.output, self.gt)
        else:
            self.net_g.eval()
            with torch.no_grad():
                index = self.opt['network_g']
                net = index['type']
                if str(net) == 'UNetArch':
                    self.output = self.net_g(self.lq)

                else:
                    self.output = self.net_g(self.lq, self.ratio)
                # illumination correction
                if self.correct:
                    self.output = self.corrector(self.output, self.gt)
            self.net_g.train()

        self.output = self.output[:, :, :h, :w]
        self.lq = self.lq[:, :, :h, :w]
        self.gt = self.gt[:, :, :h, :w]

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    @property
    def calculate_metric_in_batch(self):
        if hasattr(self, '_calculate_metric_in_batch'):
            return self._calculate_metric_in_batch

        ## init
        self._calculate_metric_in_batch = False
        if self.opt['val'].get('calculate_metric_in_batch', False) is True:
            self._calculate_metric_in_batch = True
            return self._calculate_metric_in_batch
        keys = filter(lambda x: x.startswith('val'), list(self.opt['datasets'].keys()))
        for key in keys:
            if self.opt['datasets'][key].get('batch_size_per_gpu', 1) > 1:
                self._calculate_metric_in_batch = True
                return self._calculate_metric_in_batch
        return self._calculate_metric_in_batch

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', True)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
        
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        if self.calculate_metric_in_batch:
            count = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals(self.metric_in_srgb, save_img)
            if not self.calculate_metric_in_batch:
                sr_img = tensor2img([visuals['result']])
                metric_data['img'] = sr_img
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
            else:
                metric_data['img'] = visuals['result']
                metric_data['img2'] = visuals['gt']
                count += visuals['gt'].shape[0]
            del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.ccm
            del self.wb
            torch.cuda.empty_cache()

            psnr = None
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if self.calculate_metric_in_batch and not opt_['type'].endswith('_pt'):
                        opt_['type'] = opt_['type'] + '_pt'
                    metric = calculate_metric(metric_data, opt_)
                    if self.calculate_metric_in_batch:
                        metric = torch.sum(metric)
                    self.metric_results[name] += metric
                    if name == 'psnr':
                        psnr = metric
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

            if save_img:
                if not self.calculate_metric_in_batch:
                    if not self.metric_in_srgb:
                        sr_img = tensor2img([visuals['result_srgb']])
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}.jpg')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}_{self.opt["val"]["suffix"]}.jpg')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}_{self.opt["name"]}.jpg')
                    imwrite(sr_img, save_img_path)
                else:
                    if not self.metric_in_srgb:
                        sr_imgs = tensor2img(visuals['result_srgb'])
                    else:
                        sr_imgs = tensor2img(visuals['result'])
                    if len(sr_imgs.shape) == 3:
                        if self.opt['is_train']:
                            save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                    f'{img_name}_{current_iter}.jpg')
                        else:
                            if self.opt['val']['suffix']:
                                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                        f'{img_name}_{self.opt["val"]["suffix"]}_{psnr:.4f}.jpg')
                            else:
                                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                        f'{img_name}_{self.opt["name"]}_{psnr:.4f}.jpg')
                        imwrite(sr_imgs, save_img_path)
                    else:
                        raise NotImplementedError()
                        

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                if not self.calculate_metric_in_batch:
                    self.metric_results[metric] /= (idx + 1)
                else:
                    self.metric_results[metric] /= count
                    self.metric_results[metric] = self.metric_results[metric].item()
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self, isp=True, save_img=False):
        out_dict = OrderedDict()
        if isp:
            out_dict['lq'] = raw2rgb_torch(self.lq.detach(), self.wb, self.ccm, self.CRF, batch=True)
            out_dict['result'] = raw2rgb_torch(self.output.detach(), self.wb, self.ccm, self.CRF, batch=True)
            out_dict['gt'] = raw2rgb_torch(self.gt.detach(), self.wb, self.ccm, self.CRF, batch=True)
        else:
            out_dict['lq'] = self.lq.detach()
            out_dict['result'] = self.output.detach()
            out_dict['gt'] = self.gt.detach()
            if save_img:
                out_dict['result_srgb'] = raw2rgb_torch(self.output.detach(), self.wb, self.ccm, self.CRF, batch=True)
                if not self.calculate_metric_in_batch:
                    out_dict['result_srgb'] = out_dict['result_srgb'].cpu()
        if not self.calculate_metric_in_batch:
            out_dict['lq'] = out_dict['lq'].cpu()
            out_dict['result'] = out_dict['result'].cpu()
            out_dict['gt'] = out_dict['gt'].cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)






@MODEL_REGISTRY.register()
class RAWDenoiserModel(BaseModel):

    def __init__(self, opt):
        super(RAWDenoiserModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        self.blend_merten = BlendMertens(contrast_weight=1.0, saturation_weight=1.0, exposure_weight=1.0, clip=True)
        self.noise_gen = NoiseGenerator(**self.opt['noise_g'])

        network_d = build_network(opt['network_d'])
        load_network(network_d, opt['network_d_path'])
        self.mapnet = network_d.to(self.device)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.opt.get('CRF_path', None) is not None:
            self.CRF = load_CRF(self.opt['CRF_path'])
        else:
            self.CRF = None

        self.correct = self.opt['val'].get('illumination_correct', False)
        if self.correct:
            self.corrector = IlluminanceCorrect()
        self.metric_in_srgb = self.opt.get('metric_in_srgb', False)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('srgb_opt'):
            self.post_process = lambda x, wb, ccm: raw2rgb_torch_grad(x, wb, ccm, self.CRF)
        else:
            self.post_process = lambda x, wb, ccm: x

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        learnable_layers = train_opt.get('learnable_layers', None)
        learnable_keys = train_opt.get('learnable_keys', None)
        optim_params = []
        if learnable_layers is None and learnable_keys is None:
            for k, v in self.net_g.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Params {k} will not be optimized.')
        else:
            if learnable_keys is not None:
                logger = get_root_logger()
                logger.info(f'Using \'learnable_keys\' for query training paarameters ...')
                for k, v in self.net_g.named_parameters():
                    for l_key in learnable_keys:
                        if l_key in k:
                            optim_params.append(v)
                            break
                assert len(optim_params) > 0
            if learnable_layers is not None:
                logger = get_root_logger()
                logger.info(f'Using \'learnable_layers\' for query training paarameters ...')
                for layer in learnable_layers:
                    if hasattr(self.net_g, layer):
                        optim_params.extend(list(eval(f'self.net_g.{layer}').parameters()))
                    else:
                        logger = get_root_logger()
                        logger.error(f'Layer {layer} is not in {self.net_g.__name__}.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):

        self.intact = data['intact'].to(self.device)
        self.lq_clean = data['lq_clean'].to(self.device)
        self.ccm = data['ccm'].to(self.device)
        self.wb = data['wb'].to(self.device)
        self.ratio = data['ratio'].to(self.device)
        self.ratio_all = data['ratio1'].to(self.device)

        mexp_lq = data['gt'].to(self.device)
        fused_raw = torch.clamp(exposure_fusion_from_raw(mexp_lq, self.wb, self.ccm, self.blend_merten), 1e-8)
        self.ratio_all = self.ratio_all.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        # add noise
        lq_im_patch = torch.clamp(self.lq_clean , min=0) * (16383 - 512) / self.ratio_all
        im_patch, noise1 = self.noise_gen(lq_im_patch)
        lq_im_patch = sum_img_and_noise(im_patch, noise1) / (16383 - 512) * self.ratio_all

        with torch.no_grad():
            ratiomap = self.mapnet(lq_im_patch)
        self.lq = lq_im_patch / (ratiomap + 1e-8)
        self.ratiomap = self.ratio_all / (ratiomap + 1e-8)
        self.gt = fused_raw
        self.lq, self.gt = self.lq.clip(0, 1), self.gt.clip(0, 1)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        index = self.opt['network_g']
        net = index['type']

        if str(net) == 'UNetArch' or str(net) == 'Restormer':
            self.output = self.net_g(self.lq)
        else:
            self.output = self.net_g(self.lq, self.ratiomap)

        self.output = self.post_process(self.output, self.wb, self.ccm)
        self.gt = self.post_process(self.gt, self.wb, self.ccm)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:

            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        # padding
        h, w = self.lq.shape[2:]
        pad_h = 16 - (h % 16) if h % 16 != 0 else 0
        pad_w = 16 - (w % 16) if w % 16 != 0 else 0
        
        self.gt = self.gt.squeeze(0)
        self.lq = nn.functional.pad(self.lq, [0, pad_w, 0, pad_h], mode='replicate')
        self.gt = nn.functional.pad(self.gt, [0, pad_w, 0, pad_h], mode='replicate')

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
                # illumination correction
                if self.correct:
                    self.output = self.corrector(self.output, self.gt)
        else:
            self.net_g.eval()
            with torch.no_grad():
                index = self.opt['network_g']
                net = index['type']
                if str(net) == 'UNetArch':
                    self.output = self.net_g(self.lq)

                else:
                    self.output = self.net_g(self.lq, self.ratio)
                # illumination correction
                if self.correct:
                    self.output = self.corrector(self.output, self.gt)
            self.net_g.train()

        self.output = self.output[:, :, :h, :w]
        self.lq = self.lq[:, :, :h, :w]
        self.gt = self.gt[:, :, :h, :w]

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    @property
    def calculate_metric_in_batch(self):
        if hasattr(self, '_calculate_metric_in_batch'):
            return self._calculate_metric_in_batch

        ## init
        self._calculate_metric_in_batch = False
        if self.opt['val'].get('calculate_metric_in_batch', False) is True:
            self._calculate_metric_in_batch = True
            return self._calculate_metric_in_batch
        keys = filter(lambda x: x.startswith('val'), list(self.opt['datasets'].keys()))
        for key in keys:
            if self.opt['datasets'][key].get('batch_size_per_gpu', 1) > 1:
                self._calculate_metric_in_batch = True
                return self._calculate_metric_in_batch
        return self._calculate_metric_in_batch

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', True)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        if self.calculate_metric_in_batch:
            count = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals(self.metric_in_srgb, save_img)
            if not self.calculate_metric_in_batch:
                sr_img = tensor2img([visuals['result']])
                metric_data['img'] = sr_img
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
            else:
                metric_data['img'] = visuals['result']
                metric_data['img2'] = visuals['gt']
                count += visuals['gt'].shape[0]
            del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.ccm
            del self.wb
            torch.cuda.empty_cache()

            psnr = None
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if self.calculate_metric_in_batch and not opt_['type'].endswith('_pt'):
                        opt_['type'] = opt_['type'] + '_pt'
                    metric = calculate_metric(metric_data, opt_)
                    if self.calculate_metric_in_batch:
                        metric = torch.sum(metric)
                    self.metric_results[name] += metric
                    if name == 'psnr':
                        psnr = metric
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

            if save_img:
                if not self.calculate_metric_in_batch:
                    if not self.metric_in_srgb:
                        sr_img = tensor2img([visuals['result_srgb']])
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}.jpg')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}_{self.opt["val"]["suffix"]}.jpg')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}_{self.opt["name"]}.jpg')
                    imwrite(sr_img, save_img_path)
                else:
                    if not self.metric_in_srgb:
                        sr_imgs = tensor2img(visuals['result_srgb'])
                    else:
                        sr_imgs = tensor2img(visuals['result'])
                    if len(sr_imgs.shape) == 3:
                        if self.opt['is_train']:
                            save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                    f'{img_name}_{current_iter}.jpg')
                        else:
                            if self.opt['val']['suffix']:
                                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                        f'{img_name}_{self.opt["val"]["suffix"]}_{psnr:.4f}.jpg')
                            else:
                                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                        f'{img_name}_{self.opt["name"]}_{psnr:.4f}.jpg')
                        imwrite(sr_imgs, save_img_path)
                    else:
                        raise NotImplementedError()
                        

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                if not self.calculate_metric_in_batch:
                    self.metric_results[metric] /= (idx + 1)
                else:
                    self.metric_results[metric] /= count
                    self.metric_results[metric] = self.metric_results[metric].item()
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self, isp=True, save_img=False):
        out_dict = OrderedDict()
        if isp:
            out_dict['lq'] = raw2rgb_torch(self.lq.detach(), self.wb, self.ccm, self.CRF, batch=True)
            out_dict['result'] = raw2rgb_torch(self.output.detach(), self.wb, self.ccm, self.CRF, batch=True)
            out_dict['gt'] = raw2rgb_torch(self.gt.detach(), self.wb, self.ccm, self.CRF, batch=True)
        else:
            out_dict['lq'] = self.lq.detach()
            out_dict['result'] = self.output.detach()
            out_dict['gt'] = self.gt.detach()
            if save_img:
                out_dict['result_srgb'] = raw2rgb_torch(self.output.detach(), self.wb, self.ccm, self.CRF, batch=True)
                if not self.calculate_metric_in_batch:
                    out_dict['result_srgb'] = out_dict['result_srgb'].cpu()
        if not self.calculate_metric_in_batch:
            out_dict['lq'] = out_dict['lq'].cpu()
            out_dict['result'] = out_dict['result'].cpu()
            out_dict['gt'] = out_dict['gt'].cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)