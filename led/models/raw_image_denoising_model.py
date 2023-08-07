from copy import deepcopy
import os
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from led.archs import build_network
from led.data.noise_utils import VirtualNoisyPairGenerator, CalibratedNoisyPairGenerator
from led.data.noise_utils import BasicTransform
from led.archs.repnr_utils import RepNRBase, build_repnr_arch_from_base
from led.losses import build_loss
from led.models import lr_scheduler
from led.utils import get_root_logger
from led.utils.registry import MODEL_REGISTRY
from .raw_base_model import RAWBaseModel


@MODEL_REGISTRY.register()
class RAWImageDenoisingModel(RAWBaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(RAWImageDenoisingModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        repnr_opt = self.opt.get('repnr_opt', None)
        if repnr_opt:
            self.generalize = True
            logger = get_root_logger()
            logger.info(f'Convert {self.net_g._get_name()} into RepNRBase using kwargs:\n{repnr_opt}')
            self.net_g = build_repnr_arch_from_base(self.net_g, **repnr_opt)
            self.net_g.pretrain()
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        if 'noise_g' in opt:
            self.noise_g = self.build_noise_g(opt['noise_g'])
            self.noise_g = self.noise_g.to(self.device)
            self.print_network(self.noise_g)

            noise_g_path = self.opt['path'].get('predefined_noise_g', None)
            if noise_g_path is not None:
                self.load_network(self.noise_g, noise_g_path, self.opt['path'].get('strict_load_g', True), None)
            logger = get_root_logger()
            logger.info(f'Sampled Cameras: \n{self.noise_g.log_str}')

            dump_path = os.path.join(self.opt['path']['experiments_root'], 'noise_g.pth')
            torch.save(self.noise_g.state_dict(), dump_path)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def build_noise_g(self, opt):
        opt = deepcopy(opt)
        noise_g_class = eval(opt.pop('type'))
        return noise_g_class(opt, self.device)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        repnr_opt = self.opt.get('repnr_opt', None)

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if repnr_opt:
                self.net_g_ema = build_repnr_arch_from_base(self.net_g_ema, **repnr_opt)
                self.net_g_ema.pretrain()
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        if train_opt.get('augment_on_the_fly'):
            self.augment = BasicTransform(**train_opt['augment_on_the_fly'])
        else:
            self.augment = None

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            raise NotImplementedError()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        self.ccm = data['ccm'].to(self.device)
        self.wb = data['wb'].to(self.device)
        self.ratio = data['ratio'].to(self.device)
        if 'black_level' in data:
            self.black_level = data['black_level'].to(self.device)
            self.white_level = data['white_level'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        if hasattr(self, 'noise_g'):
            self.camera_id = torch.randint(0, len(self.noise_g), (1,)).item()
            with torch.no_grad():
                scale = self.white_level - self.black_level
                self.gt = (self.gt - self.black_level) / scale
                self.gt, self.lq, self.curr_metadata = self.noise_g(self.gt, scale, self.ratio, self.camera_id)
                if self.augment is not None:
                    self.gt, self.lq = self.augment(self.gt, self.lq)

        if isinstance(self.net_g, RepNRBase):
            self.output = self.net_g(self.lq, self.camera_id)
        else:
            self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        l_pix = self.cri_pix(self.output, self.gt)
        l_total += l_pix
        loss_dict['l_pix'] = l_pix

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


@MODEL_REGISTRY.register()
class LEDFinetuneModel(RAWBaseModel):
    def __init__(self, opt):
        super(LEDFinetuneModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])

        repnr_opt = opt['repnr_opt']
        self.repnr_opt = deepcopy(repnr_opt)
        logger = get_root_logger()
        logger.info(f'Convert {self.net_g._get_name()} into RepNRBase using kwargs:\n{repnr_opt}')
        self.net_g = build_repnr_arch_from_base(self.net_g, **repnr_opt)

        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        else:
            raise NotImplementedError(self.__name__ + ' is only for fintunning, please specify a pretrained path')

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.total_iter = train_opt['total_iter']
        self.align_iter = train_opt['align_iter']
        self.oomn_iter  = train_opt['oomn_iter']
        assert (self.oomn_iter + self.align_iter) == self.total_iter
        if train_opt.get('generalize_first') and hasattr(self.net_g, 'generalize'):
            self.net_g.generalize()
        self.net_g.finetune(aux=self.oomn_iter > 0)

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            self.net_g_ema = build_repnr_arch_from_base(self.net_g_ema, **deepcopy(self.repnr_opt))
            self.net_g_ema.finetune(aux=self.oomn_iter > 0)
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
            raise NotImplementedError()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        align_params, oomn_params = [], []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if '.align_weights' in k or '.align_biases' in k:
                    align_params.append(v)
                elif '.aux_weight' in k or '.aux_bias' in k:
                    oomn_params.append(v)
                else:
                    logger.warning(f'Params {k} will not be optimized, though it requires grad!')
            else:
                logger.warning(f'Params {k} will not be optimized.')

        optim_align_type = train_opt['align_opt']['optim_g'].pop('type')
        self.optimizer_align = self.get_optimizer(optim_align_type, align_params, **train_opt['align_opt']['optim_g'])
        self.optimizers.append(self.optimizer_align)
        self.cur_optimizer = self.optimizer_align

        if self.oomn_iter > 0:
            optim_oomn_type  = train_opt['oomn_opt']['optim_g'].pop('type')
            self.optimizer_oomn = self.get_optimizer(optim_oomn_type, oomn_params, **train_opt['oomn_opt']['optim_g'])
            self.optimizers.append(self.optimizer_oomn)

    def setup_schedulers(self):
        def get_scheduler_class(scheduler_type):
            if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
                return lr_scheduler.MultiStepRestartLR
            elif scheduler_type == 'CosineAnnealingRestartLR':
                return lr_scheduler.CosineAnnealingRestartLR
            elif scheduler_type == 'HandieLR':
                return lr_scheduler.HandieLR
            else:
                raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

        """Set up schedulers."""
        train_opt = self.opt['train']
        optim_align_type = train_opt['align_opt']['scheduler'].pop('type')
        self.scheduler_align = get_scheduler_class(optim_align_type)(self.optimizer_align,
                                                                     **train_opt['align_opt']['scheduler'])
        self.schedulers.append(self.scheduler_align)
        self.cur_scheduler = self.scheduler_align

        if self.oomn_iter > 0:
            optim_oomn_type  = train_opt['oomn_opt']['scheduler'].pop('type')
            self.scheduler_oomn = get_scheduler_class(optim_oomn_type)(self.optimizer_oomn,
                                                                       **train_opt['oomn_opt']['scheduler'])
            self.schedulers.append(self.scheduler_oomn)

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warm-up iter numbers. -1 for no warm-up.
                Default： -1.
        """
        if current_iter > 1:
            self.cur_scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.cur_optimizer.param_groups]

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        self.ccm = data['ccm'].to(self.device)
        self.wb = data['wb'].to(self.device)
        self.ratio = data['ratio'].to(self.device)
        if 'black_level' in data:
            self.black_level = data['black_level'].to(self.device)
            self.white_level = data['white_level'].to(self.device)

    def optimize_parameters(self, current_iter):
        if current_iter == (self.align_iter + 1):
            logger = get_root_logger()
            logger.info('Switch to optimize oomn branch....')
            self.cur_optimizer = self.optimizer_oomn
            self.cur_scheduler = self.scheduler_oomn
        self.cur_optimizer.zero_grad()

        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        l_pix = self.cri_pix(self.output, self.gt)
        l_total += l_pix
        loss_dict['l_pix'] = l_pix

        l_total.backward()

        self.cur_optimizer.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            if isinstance(self.net_g_ema, RepNRBase):
                self.save_network([
                    self.net_g, self.net_g.deploy(),
                    self.net_g_ema, self.net_g_ema.deploy()
                ], 'net_g', current_iter, param_key=['params', 'params_deploy', 'params_ema', 'params_ema_deploy'])
            else:
                self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            if isinstance(self.net_g, RepNRBase):
                self.save_network([
                    self.net_g, self.net_g.deploy()
                ], 'net_g', current_iter, param_key=['params', 'params_deploy'])
            else:
                self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
