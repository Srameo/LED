import logging
import torch
from os import path as osp
from glob import glob

from ultraled.data import build_dataloader, build_dataset
from ultraled.models import build_model
from ultraled.train import init_tb_loggers
from ultraled.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from ultraled.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)
    opt['root_path'] = root_path
    experiments_root = osp.join(root_path, 'results', opt['name'])
    opt['path']['experiments_root'] = experiments_root
    opt['path']['log'] = experiments_root
    opt['path']['visualization'] = osp.join(experiments_root, 'visualization')

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    paths = glob(f"{opt['path']['pretrain_network_g_dir']}/*.pth")
    if f"{opt['path']['pretrain_network_g_dir']}/net_g_latest.pth" in paths:
        paths.pop(paths.index(f"{opt['path']['pretrain_network_g_dir']}/net_g_latest.pth"))
    paths = list(sorted(paths, key=lambda x: int(x[:-4].split('_')[-1])))
    opt['path']['pretrain_network_g_dir'] = None
    # create model
    model = build_model(opt)

    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)
    for load_path in paths:
        if load_path.endswith('net_g_latest.pth'):
            continue
        param_key = opt['path'].get('param_key_g', 'params')
        model.load_network(model.net_g, load_path, opt['path'].get('strict_load_g', True), param_key)
        current_iter = int(load_path[:-4].split('_')[-1])
        for test_loader in test_loaders:
            test_set_name = test_loader.dataset.opt['name']
            logger.info(f'Testing {test_set_name}...')
            model.validation(test_loader, current_iter=current_iter, tb_logger=tb_logger, save_img=opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
