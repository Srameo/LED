import math
from numbers import Number
import random
from copy import deepcopy
import pandas as pd
from tabulate import tabulate
import torch
from torch import nn
import numpy as np
from .common import *

def _uniform_batch(min_, max_, shape=(1,), device='cpu'):
    return torch.rand(shape, device=device) * (max_ - min_) + min_

def _normal_batch(scale=1.0, loc=0.0, shape=(1,), device='cpu'):
    return torch.randn(shape, device=device) * scale + loc

def _randint_batch(min_, max_, shape=(1,), device='cpu'):
    return torch.randint(min_, max_, shape, device=device)


class CalibratedNoisyPairGenerator(nn.Module):
    def __init__(self, opt, device='cuda') -> None:
        super().__init__()
        self.device = device
        self.opt = deepcopy(opt)
        self.camera_params = opt['camera_params']
        self.camera_params = self.param_dict_to_tensor_dict(self.camera_params)
        self.cameras = list(self.camera_params.keys())
        print('Current Using Cameras: ', self.cameras)

        self.noise_type = opt['noise_type'].lower()
        self.read_type = 'TukeyLambda' if 't' in self.noise_type else \
            ('Gaussian' if 'g' in self.noise_type else None)

    def param_dict_to_tensor_dict(self, p_dict):
        def to_tensor_dict(p_dict):
            for k, v in p_dict.items():
                if isinstance(v, list) or isinstance(v, Number):
                    p_dict[k] = nn.Parameter(torch.tensor(v, device=self.device), False)
                elif isinstance(v, dict):
                    p_dict[k] = to_tensor_dict(v)
            return p_dict
        return to_tensor_dict(p_dict)

    def sample_overall_system_gain(self, batch_size, for_video):
        if self.index is None:
            self.index = _randint_batch(0, len(self.camera_params)).item()
        self.current_camera = self.cameras[self.index]
        self.current_camera_params = self.camera_params[self.current_camera]
        self.current_k_range = [
            self.camera_params[self.current_camera]['Kmin'],
            self.camera_params[self.current_camera]['Kmax']
        ]
        log_K_max = torch.log(self.current_camera_params['Kmax'])
        log_K_min = torch.log(self.current_camera_params['Kmin'])
        log_K = _uniform_batch(log_K_min, log_K_max, (batch_size, 1, 1, 1), self.device)
        if for_video:
            log_K = log_K.unsqueeze(-1)
        self.log_K = log_K
        self.cur_batch_size = batch_size
        return torch.exp(log_K)

    def sample_read_sigma(self):
        slope = self.current_camera_params[self.read_type]['slope']
        bias = self.current_camera_params[self.read_type]['bias']
        sigma = self.current_camera_params[self.read_type]['sigma']
        mu = self.log_K.squeeze() * slope + bias
        sample = _normal_batch(sigma, mu, (self.cur_batch_size,), self.device)
        return torch.exp(sample).reshape(self.log_K.shape)

    def sample_tukey_lambda(self, batch_size, for_video):
        index = _randint_batch(0, len(self.current_camera_params[self.read_type]['lambda']), shape=(batch_size,))
        tukey_lambdas = self.current_camera_params[self.read_type]['lambda'][index].reshape(batch_size, 1, 1, 1)
        if for_video:
            tukey_lambdas = tukey_lambdas.unsqueeze(1)
        return tukey_lambdas

    def sample_row_sigma(self):
        slope = self.current_camera_params['Row']['slope']
        bias = self.current_camera_params['Row']['bias']
        sigma = self.current_camera_params['Row']['sigma']
        mu = self.log_K.squeeze() * slope + bias
        sample = _normal_batch(sigma, mu, (self.cur_batch_size,), self.device)
        return torch.exp(sample).reshape(self.log_K.squeeze(-3).shape)

    def sample_color_bias(self, batch_size, for_video):
        count = len(self.current_camera_params['ColorBias'])
        i_range = (self.current_k_range[1] - self.current_k_range[0]) / count
        index = ((torch.exp(self.log_K.squeeze()) - self.current_k_range[0]) // i_range).long()
        color_bias = self.current_camera_params['ColorBias'][index]
        color_bias = color_bias.reshape(batch_size, 4, 1, 1)
        if for_video:
            color_bias = color_bias.unsqueeze(1)
        return color_bias

    @staticmethod
    def add_noise(img, noise, noise_params, if_clip=True):
        tail = [1 for _ in range(img.dim() - 1)]
        ratio = noise_params['isp_dgain'].view(-1, *tail)
        scale = noise_params['scale'].view(-1, 4, *tail[:-1])
        for n in noise.values():
            img += n
        img /= scale
        img = img * ratio
        if if_clip:
            img = torch.clamp(img, max=1.0)
        return img

    @torch.no_grad()
    def forward(self, img, scale, ratio, vcam_id=None, if_clip=True):
        b = img.size(0)
        for_video = True if img.dim() == 5 else False # B, T, C, H, W
        self.index = vcam_id if vcam_id is not None else None

        if if_clip:
            img_gt = torch.clamp(img, 0, 1)
        else:
            img_gt = img
        tail = [1 for _ in range(img.dim() - 1)]
        img = img_gt * scale.view(-1, 4, *tail[:-1]) / ratio.view(-1, *tail)

        K = self.sample_overall_system_gain(b, for_video)
        noise = {}
        noise_params = {'isp_dgain': ratio, 'scale': scale}
        # shot noise
        if 'p' in self.noise_type:
            _shot_noise = shot_noise(img, K)
            noise['shot'] = _shot_noise
            noise_params['shot'] = K.squeeze()
        # read noise
        if 'g' in self.noise_type:
            read_param = self.sample_read_sigma()
            _read_noise = gaussian_noise(img, read_param)
            noise['read'] = _read_noise
            noise_params['read'] = read_param.squeeze()
        elif 't' in self.noise_type:
            tukey_lambda = self.sample_tukey_lambda(b, for_video)
            read_param = self.sample_read_sigma()
            _read_noise = tukey_lambda_noise(img, read_param, tukey_lambda)
            noise['read'] = _read_noise
            noise_params['read'] = {
                'sigma': read_param,
                'tukey_lambda': tukey_lambda
            }
        # row noise
        if 'r' in self.noise_type:
            row_param = self.sample_row_sigma()
            _row_noise = row_noise(img, row_param)
            noise['row'] = _row_noise
            noise_params['row'] = row_param.squeeze()
        # quant noise
        if 'q' in self.noise_type:
            _quant_noise = quant_noise(img, 1)
            noise['quant'] = _quant_noise
        # color bias
        if 'c' in self.noise_type:
            color_bias = self.sample_color_bias(b, for_video)
            noise['color_bias'] = color_bias

        img_lq = self.add_noise(img, noise, noise_params, if_clip=if_clip)

        return img_gt, img_lq, {
            'cam': self.current_camera,
            'noise': noise,
            'noise_params': noise_params
        }

    def __len__(self):
        return len(self.cameras)

    def cpu(self):
        super().cpu()
        self.device = 'cpu'
        return self

    def cuda(self, device=None):
        super().cuda(device)
        self.device = 'cuda'
        return self

    @property
    def log_str(self):
        return f'{self._get_name()}: {self.cameras}'


class VirtualNoisyPairGenerator(nn.Module):
    def __init__(self, opt, device='cuda') -> None:
        super().__init__()
        self.opt = deepcopy(opt)
        self.device = device
        self.sample_virtual_cameras()

        print('Current Using Cameras: ', [f'IC{i}' for i in range(self.virtual_camera_count)])

    def sample_virtual_cameras(self):
        self.noise_type = self.opt['noise_type'].lower()
        self.param_ranges = self.opt['param_ranges']
        self.virtual_camera_count = self.opt['virtual_camera_count']
        self.sample_strategy = self.opt['sample_strategy']
        self.shuffle = self.opt.get('shuffle', False)

        # sampling strategy
        sample = self.split_range if self.sample_strategy == 'coverage' else self.uniform_range

        # overall system gain
        self.k_range = torch.tensor(self.param_ranges['K'], device=self.device)

        # read noise
        if 'g' in self.noise_type:
            read_slope_range = self.param_ranges['Gaussian']['slope']
            read_bias_range = self.param_ranges['Gaussian']['bias']
            read_sigma_range = self.param_ranges['Gaussian']['sigma']
        elif 't' in self.noise_type:
            read_slope_range = self.param_ranges['TukeyLambda']['slope']
            read_bias_range = self.param_ranges['TukeyLambda']['bias']
            read_sigma_range = self.param_ranges['TukeyLambda']['sigma']
            read_lambda_range = self.param_ranges['TukeyLambda']['lambda']
            self.tukey_lambdas = sample(self.virtual_camera_count, read_lambda_range, self.shuffle, self.device)
            self.tukey_lambdas = nn.Parameter(self.tukey_lambdas, False)
        if 'g' in self.noise_type or 't' in self.noise_type:
            self.read_slopes = sample(self.virtual_camera_count, read_slope_range, self.shuffle, self.device)
            self.read_biases = sample(self.virtual_camera_count, read_bias_range, self.shuffle, self.device)
            self.read_sigmas = sample(self.virtual_camera_count, read_sigma_range, self.shuffle, self.device)
            self.read_slopes = nn.Parameter(self.read_slopes, False)
            self.read_biases = nn.Parameter(self.read_biases, False)
            self.read_sigmas = nn.Parameter(self.read_sigmas, False)

        # row noise
        if 'r' in self.noise_type:
            row_slope_range = self.param_ranges['Row']['slope']
            row_bias_range = self.param_ranges['Row']['bias']
            row_sigma_range = self.param_ranges['Row']['sigma']
            self.row_slopes = sample(self.virtual_camera_count, row_slope_range, self.shuffle, self.device)
            self.row_biases = sample(self.virtual_camera_count, row_bias_range, self.shuffle, self.device)
            self.row_sigmas = sample(self.virtual_camera_count, row_sigma_range, self.shuffle, self.device)
            self.row_slopes = nn.Parameter(self.row_slopes, False)
            self.row_biases = nn.Parameter(self.row_biases, False)
            self.row_sigmas = nn.Parameter(self.row_sigmas, False)

        # color bias
        if 'c' in self.noise_type:
            self.color_bias_count = self.param_ranges['ColorBias']['count']
            ## ascend sigma
            color_bias_sigmas = self.split_range_overlap(self.color_bias_count,
                                                         self.param_ranges['ColorBias']['sigma'],
                                                         overlap=0.1)
            self.color_biases = torch.tensor(np.array([
                [
                    random.uniform(*self.param_ranges['ColorBias']['bias']) + \
                        torch.randn(4).numpy() * random.uniform(*color_bias_sigmas[i]).cpu().numpy()
                    for _ in range(self.color_bias_count)
                ] for i in range(self.virtual_camera_count)
            ]), device=self.device)
            self.color_biases = nn.Parameter(self.color_biases, False)

    @staticmethod
    def uniform_range(splits, range_, shuffle=True, device='cuda'):
        results = [random.uniform(*range_) for _ in range(splits)]
        if shuffle:
            random.shuffle(results)
        return torch.tensor(results, device=device)

    @staticmethod
    def split_range(splits, range_, shuffle=True, device='cuda'):
        length = range_[1] - range_[0]
        i_length = length / (splits - 1)
        results = [range_[0] + i_length * i for i in range(splits)]
        if shuffle:
            random.shuffle(results)
        return torch.tensor(results, device=device)

    @staticmethod
    def split_range_overlap(splits, range_, overlap=0.5, device='cuda'):
        length = range_[1] - range_[0]
        i_length = length / (splits * (1 - overlap) + overlap)
        results = []
        for i in range(splits):
            start = i_length * (1 - overlap) * i
            results.append([start, start + i_length])
        return torch.tensor(results, device=device)

    def sample_overall_system_gain(self, batch_size, for_video):
        if self.current_camera is None:
            index = _randint_batch(0, self.virtual_camera_count, (batch_size,), self.device)
            self.current_camera = index
        log_K_max = torch.log(self.k_range[1])
        log_K_min = torch.log(self.k_range[0])
        log_K = _uniform_batch(log_K_min, log_K_max, (batch_size, 1, 1, 1), self.device)
        if for_video:
            log_K = log_K.unsqueeze(-1)
        self.log_K = log_K
        self.cur_batch_size = batch_size
        return torch.exp(log_K)

    def sample_read_sigma(self):
        slope = self.read_slopes[self.current_camera]
        bias = self.read_biases[self.current_camera]
        sigma = self.read_sigmas[self.current_camera]
        mu = self.log_K.squeeze() * slope + bias
        sample = _normal_batch(sigma, mu, (self.cur_batch_size,), self.device)
        return torch.exp(sample).reshape(self.log_K.shape)

    def sample_tukey_lambda(self, batch_size, for_video):
        tukey_lambda = self.tukey_lambdas[self.current_camera].reshape(batch_size, 1, 1, 1)
        if for_video:
            tukey_lambda = tukey_lambda.unsqueeze(-1)
        return tukey_lambda

    def sample_row_sigma(self):
        slope = self.row_slopes[self.current_camera]
        bias = self.row_biases[self.current_camera]
        sigma = self.row_sigmas[self.current_camera]
        mu = self.log_K.squeeze() * slope + bias
        sample = _normal_batch(sigma, mu, (self.cur_batch_size,), self.device)
        return torch.exp(sample).reshape(self.log_K.squeeze(-3).shape)

    def sample_color_bias(self, batch_size, for_video):
        i_range = (self.k_range[1] - self.k_range[0]) / self.color_bias_count
        index = ((torch.exp(self.log_K.squeeze()) - self.k_range[0]) // i_range).long()
        color_bias = self.color_biases[self.current_camera, index]
        color_bias = color_bias.reshape(batch_size, 4, 1, 1)
        if for_video:
            color_bias = color_bias.unsqueeze(1)
        return color_bias

    @staticmethod
    def add_noise(img, noise, noise_params):
        tail = [1 for _ in range(img.dim() - 1)]
        ratio = noise_params['isp_dgain'].view(-1, *tail)
        scale = noise_params['scale'].view(-1, 4, *tail[:-1])
        for n in noise.values():
            img += n
        img /= scale
        img = img * ratio
        return torch.clamp(img, max=1.0)

    @torch.no_grad()
    def forward(self, img, scale, ratio, vcam_id=None):
        b = img.size(0)
        for_video = True if img.dim() == 5 else False # B, T, C, H, W
        self.current_camera = vcam_id * torch.ones((b,), dtype=torch.long, device=self.device) \
                              if vcam_id is not None else None

        img_gt = torch.clamp(img, 0, 1)
        tail = [1 for _ in range(img.dim() - 1)]
        img = img_gt * scale.view(-1, 4, *tail[:-1]) / ratio.view(-1, *tail)

        K = self.sample_overall_system_gain(b, for_video)
        noise = {}
        noise_params = {'isp_dgain': ratio, 'scale': scale}
        # shot noise
        if 'p' in self.noise_type:
            _shot_noise = shot_noise(img, K)
            noise['shot'] = _shot_noise
            noise_params['shot'] = K.squeeze()
        # read noise
        if 'g' in self.noise_type:
            read_param = self.sample_read_sigma()
            _read_noise = gaussian_noise(img, read_param)
            noise['read'] = _read_noise
            noise_params['read'] = read_param.squeeze()
        elif 't' in self.noise_type:
            tukey_lambda = self.sample_tukey_lambda(b, for_video)
            read_param = self.sample_read_sigma()
            _read_noise = tukey_lambda_noise(img, read_param, tukey_lambda)
            noise['read'] = _read_noise
            noise_params['read'] = {
                'sigma': read_param,
                'tukey_lambda': tukey_lambda
            }
        # row noise
        if 'r' in self.noise_type:
            row_param = self.sample_row_sigma()
            _row_noise = row_noise(img, row_param)
            noise['row'] = _row_noise
            noise_params['row'] = row_param.squeeze()
        # quant noise
        if 'q' in self.noise_type:
            _quant_noise = quant_noise(img, 1)
            noise['quant'] = _quant_noise
        # color bias
        if 'c' in self.noise_type:
            color_bias = self.sample_color_bias(b, for_video)
            noise['color_bias'] = color_bias

        img_lq = self.add_noise(img, noise, noise_params)

        return img_gt, img_lq, {
            'vcam_id': self.current_camera.squeeze(),
            'noise': noise,
            'noise_params': noise_params
        }

    def __len__(self):
        return self.virtual_camera_count

    def cpu(self):
        super().cpu()
        self.device = self.k_range.device
        return self

    def cuda(self, device=None):
        super().cuda(device)
        self.device = self.k_range.device
        return self

    @property
    def json_dict(self):
        if hasattr(self, '_json_dict'):
            return self._json_dict

        json_dict = { f'IC{i}': {} for i in range(self.virtual_camera_count) }
        for i in range(self.virtual_camera_count):
            json_dict[f'IC{i}']['Kmin'] = self.k_range[0].cpu().numpy().tolist()
            json_dict[f'IC{i}']['Kmax'] = self.k_range[1].cpu().numpy().tolist()

        if 'g' in self.noise_type or 't' in self.noise_type:
            read_log_key = 'G' if 'g' in self.noise_type else 'TL'
            for i in range(len(self.read_slopes)):
                json_dict[f'IC{i}'][f'{read_log_key}_slope'] = self.read_slopes[i].cpu().numpy().tolist()
                json_dict[f'IC{i}'][f'{read_log_key}_bias'] = self.read_biases[i].cpu().numpy().tolist()
                json_dict[f'IC{i}'][f'{read_log_key}_sigma'] = self.read_sigmas[i].cpu().numpy().tolist()
                if read_log_key == 'TL':
                    json_dict[f'IC{i}'][f'{read_log_key}_lambda'] = self.tukey_lambdas[i].cpu().numpy().tolist()

        if 'r' in self.noise_type:
            for i in range(len(self.row_slopes)):
                json_dict[f'IC{i}']['Row_slope'] = self.row_slopes[i].cpu().numpy().tolist()
                json_dict[f'IC{i}']['Row_bias'] = self.row_biases[i].cpu().numpy().tolist()
                json_dict[f'IC{i}']['Row_sigma'] = self.row_sigmas[i].cpu().numpy().tolist()

        if 'c' in self.noise_type:
            for i in range(len(self.color_biases)):
                json_dict[f'IC{i}']['CB_biases'] = self.color_biases[i].cpu().numpy().tolist()

        self._json_dict = json_dict
        return json_dict

    @property
    def log_str(self):
        def clip_float_in_list(l, fmt=4, auto_newline=True):
            l_out = '['
            count = len(l)
            for i, f in enumerate(l):
                if torch.is_tensor(f):
                    f = f.cpu().numpy()
                if auto_newline and i % int(math.sqrt(count)) == 0 and not isinstance(f, np.ndarray):
                    l_out += '\n  '
                if isinstance(f, np.ndarray):
                    l_out += '\n  ' + str(np.array(f * 10 ** fmt, dtype='int') / float(10 ** fmt)) + ','
                else:
                    l_out += str(int(f * 10 ** fmt) / float(10 ** fmt)) + ', '
            else:
                if auto_newline:
                    l_out += '\n'
            l_out += ']'
            return l_out

        color_biases = deepcopy(self.color_biases)
        json_dict = deepcopy(self.json_dict)
        if 'c' in self.noise_type:
            for i in range(len(color_biases)):
                json_dict[f'IC{i}']['CB_biases'] = clip_float_in_list(color_biases[i])
        return tabulate(pd.DataFrame(json_dict).T, headers="keys", floatfmt='.4f')
