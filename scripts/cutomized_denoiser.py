import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tag', type=str, required=True, help='the name for the test.')
    parser.add_argument('-p', '--pretrained_network', type=str, required=True, help='the pretrained network path.')
    parser.add_argument('--dataroot', type=str, required=True, help='the dataroot for your captured train images.')
    parser.add_argument('--data_pair_list', type=str, required=True, help='the data list for check your image pairs.')
    parser.add_argument(
        '--force_yml', nargs='+', default=[], help='Force to update yml files. Examples: train:ema_decay=0.999')
    args = parser.parse_args()

    command_list = []
    base_command = 'python led/train.py -opt options/base/demo/base.yaml --force_yml'
    command_list.append(base_command)

    name_command = f'name={args.tag}'
    command_list.append(name_command)

    pretrained_network_command = f'path:pretrain_network_g={args.pretrained_network}'
    command_list.append(pretrained_network_command)

    dataroot_command = f'datasets:train:dataroot={args.dataroot}'
    command_list.append(dataroot_command)

    data_pair_list_command = f'datasets:train:data_pair_list={args.data_pair_list}'
    command_list.append(data_pair_list_command)

    command_list.extend(args.force_yml)
    command = ' '.join(command_list)
    os.system(command)
