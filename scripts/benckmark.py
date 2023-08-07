import os
import argparse

DATASET_HASH_LIST = {
    'SID': {
        'SonyA7S2': 'options/base/dataset/test/SID_SonyA7S2_val_split.yaml'
    },
    'ELD': {
        'SonyA7S2': 'options/base/dataset/test/ELD_SonyA7S2_val_split.yaml',
        'NikonD850': 'options/base/dataset/test/ELD_NikonD850_val_split.yaml'
    }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tag', type=str, required=True, help='the name for the test.')
    parser.add_argument('-p', '--pretrained_network', type=str, required=True, help='the pretrained network path.')
    parser.add_argument(
        '--dataset', nargs='+', default=['SID', 'SonyA7S2'], help='a two element list for define test dataset.')
    parser.add_argument('--save_img', action='store_true')
    parser.add_argument('--led', action='store_true')
    args = parser.parse_args()

    command_list = []
    base_command = 'python led/test.py -opt options/base/test/base.yaml --force_yml'
    command_list.append(base_command)

    name_command = f'name={args.tag}'
    command_list.append(name_command)

    assert len(args.dataset) == 2
    dataset_command = f'base:1={DATASET_HASH_LIST[args.dataset[0]][args.dataset[1]]}'
    command_list.append(dataset_command)

    pretrained_network_command = f'path:pretrain_network_g={args.pretrained_network}'
    command_list.append(pretrained_network_command)

    save_img_command = f'val:save_img={args.save_img}'
    command_list.append(save_img_command)

    led_command = f'path:param_key_g=\'params_deploy\'' if args.led else ''
    command_list.append(led_command)

    command = ' '.join(command_list)
    os.system(command)
