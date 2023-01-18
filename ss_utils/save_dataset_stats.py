"""
Go through entire dataset to obtain and save
mean, std, min, max of hand x, y, depth, orientation, contact
(used as training baseline loss and data normalization)
"""
import json
import os
import os.path as osp
import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ss_utils.filter_utils import determine_which_hand

DEPTH_DESC = 'scaling_factor'
DEPTH_PARAMS_PATH = '/home/junyao/LfHV/frankmocap/ss_utils/depth_normalization_params.pkl'
ORI_PARAMS_PATH = '/home/junyao/LfHV/frankmocap/ss_utils/ori_normalization_params.pkl'
DATA_PARAMS_SAVE_PATH = '/home/junyao/LfHV/frankmocap/ss_utils/data_params.pkl'


def process_mocap_paths(mocap_paths):
    data_dict = {feat: [] for feat in ['x', 'y', 'contact']}
    for mocap_path in tqdm(mocap_paths, desc=f'Going through {len(mocap_paths)} frames.'):
        with open(mocap_path, 'rb') as f:
            hand_info = pickle.load(f)
        hand = determine_which_hand(hand_info)

        img_shape = hand_info['image_shape']
        img_x, img_y = img_shape[1], img_shape[0]
        wrist_3d = hand_info['pred_output_list'][0][hand]['pred_joints_img'][0]
        contact = hand_info['contact_filtered']
        wrist_x_float, wrist_y_float = wrist_3d[:2]
        wrist_x_normalized = wrist_x_float / float(img_x)
        wrist_y_normalized = wrist_y_float / float(img_y)

        data_dict['x'].append(wrist_x_normalized)
        data_dict['y'].append(wrist_y_normalized)
        data_dict['contact'].append(contact)

    return data_dict


def save_dataset_stats(depth_descriptor, depth_params_path, ori_params_path, data_params_save_path):
    mocap_paths = []
    ss_dir = '/home/junyao/Datasets/something_something_processed'
    tasks = [
        'move_away',
        'move_towards',
        'move_down',
        'move_up',
        'pull_left',
        'pull_right',
        'push_left',
        'push_right',
    ]
    splits = [
        'train',
        'valid'
    ]
    task_dirs = [osp.join(ss_dir, t) for t in tasks]
    for task_dir in tqdm(task_dirs, desc='Extracting valid mocap paths from dataset dir'):
        for split in splits:
            split_dir = osp.join(task_dir, split)
            iou_json_path = [osp.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.json')][0]
            with open(iou_json_path, 'r') as f:
                iou_json_dict = json.load(f)
            for vid_num in iou_json_dict:
                mocap_paths.extend(
                    [osp.join(split_dir, 'mocap_output', vid_num, 'mocap', f'frame{frame_num}_prediction_result.pkl')
                     for frame_num in iou_json_dict[vid_num]]
                )

    data_dict = process_mocap_paths(mocap_paths)
    depth_name = f'depth_{DEPTH_DESC}'
    data_params = {v: {} for v in ['x', 'y', depth_name, 'ori', 'contact']}
    for feat, data in data_dict.items():
        data_params[feat]['min'] = np.min(data)
        data_params[feat]['max'] = np.max(data)
        data_params[feat]['mean'] = np.mean(data)
        data_params[feat]['std'] = np.std(data)
        plt.hist(data, bins=50)
        plt.title(f'{feat}')
        plt.savefig(f'data_params_plots/{feat}_hist.png')
        plt.close()

    with open(depth_params_path, 'rb') as f:
        depth_params = pickle.load(f)
    data_params[depth_name]['min'] = depth_params[depth_descriptor]['min']
    data_params[depth_name]['max'] = depth_params[depth_descriptor]['max']
    data_params[depth_name]['mean'] = depth_params[depth_descriptor]['mean']
    data_params[depth_name]['std'] = depth_params[depth_descriptor]['std']
    data_params[depth_name]['scale'] = depth_params[depth_descriptor]['scale']

    with open(ori_params_path, 'rb') as f:
        ori_params = pickle.load(f)
    data_params['ori']['min'] = ori_params['min']
    data_params['ori']['max'] = ori_params['max']
    data_params['ori']['mean'] = ori_params['mean']
    data_params['ori']['std'] = ori_params['std']

    with open(data_params_save_path, 'wb') as f:
        pickle.dump(data_params, f)


if __name__ == '__main__':
    save_dataset_stats(
        depth_descriptor=DEPTH_DESC,
        depth_params_path=DEPTH_PARAMS_PATH,
        ori_params_path=ORI_PARAMS_PATH,
        data_params_save_path=DATA_PARAMS_SAVE_PATH
    )
