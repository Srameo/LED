
from collections import OrderedDict
from os import path as osp
import torch
from tqdm import tqdm
from led.metrics import calculate_metric

from led.models.raw_denoising_utils import IlluminanceCorrect
from led.utils import get_root_logger, imwrite, tensor2img
from led.utils.process import load_CRF, raw2rgb_torch
from .base_model import BaseModel


class RAWBaseModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        CRF_path = self.opt['path'].get('CRF', None)
        self.CRF = None if CRF_path is None else load_CRF(self.opt['path']['CRF'])

        correct = False if 'val' not in self.opt else self.opt['val'].get('illumination_correct', False)
        self.corrector = IlluminanceCorrect() if correct else None

        self.metric_in_srgb = False if 'val' not in self.opt else self.opt['val'].get('metric_in_srgb', False)
        self.generalize = False

    def resume_training(self, resume_state):
        if hasattr(self, 'noise_g'):
            noise_g_path = osp.join(self.opt['path']['experiments_root'], 'noise_g.pth')
            self.load_network(self.noise_g, noise_g_path, self.opt['path'].get('strict_load_g', True), None)
        return super().resume_training(resume_state)

    @property
    def calculate_metric_in_batch(self):
        if hasattr(self, '_calculate_metric_in_batch'):
            return self._calculate_metric_in_batch

        ## init
        self._calculate_metric_in_batch = False
        if self.opt['val'].get('calculate_metric_in_batch', False):
            self._calculate_metric_in_batch = True
            return self._calculate_metric_in_batch
        keys = filter(lambda x: x.startswith('val'), list(self.opt['datasets'].keys()))
        for key in keys:
            if self.opt['datasets'][key].get('batch_size_per_gpu', 1) > 1:
                self._calculate_metric_in_batch = True
                return self._calculate_metric_in_batch
        return self._calculate_metric_in_batch

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
                if self.corrector is not None:
                    self.output = self.corrector(self.output, self.gt)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
                if self.corrector is not None:
                    self.output = self.corrector(self.output, self.gt)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', True)

        if hasattr(self.net_g, 'generalize') and self.generalize:
            self.net_g.generalize()
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.generalize()

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

            if save_img:
                if not self.calculate_metric_in_batch:
                    if not self.metric_in_srgb:
                        sr_img = tensor2img([visuals['result_srgb']])
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}_{self.opt["name"]}.png')
                    imwrite(sr_img, save_img_path)
                else:
                    if not self.metric_in_srgb:
                        sr_imgs = tensor2img(visuals['result_srgb'])
                    else:
                        sr_imgs = tensor2img(visuals['result'])
                    if len(sr_imgs.shape) == 3:
                        if self.opt['is_train']:
                            save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                    f'{img_name}_{current_iter}.png')
                        else:
                            if self.opt['val']['suffix']:
                                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                        f'{img_name}_{self.opt["val"]["suffix"]}.png')
                            else:
                                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                        f'{img_name}_{self.opt["name"]}.png')
                        imwrite(sr_imgs, save_img_path)
                    else:
                        raise NotImplementedError()

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if self.calculate_metric_in_batch and not opt_['type'].endswith('_pt'):
                        opt_['type'] = opt_['type'] + '_pt'
                    metric = calculate_metric(metric_data, opt_)
                    if self.calculate_metric_in_batch:
                        metric = torch.sum(metric)
                    self.metric_results[name] += metric
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
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
