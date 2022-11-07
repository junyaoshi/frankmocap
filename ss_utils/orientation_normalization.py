import pickle
import os.path as osp
import os
import json
from pprint import pprint

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from ss_utils.filter_utils import determine_which_hand


def get_all_oris(mocap_paths):
    oris = []
    for mocap_path in tqdm(mocap_paths, desc=f'Going through {len(mocap_paths)} frames.'):
        with open(mocap_path, 'rb') as f:
            hand_info = pickle.load(f)
        hand = determine_which_hand(hand_info)
        hand_pose = hand_info['pred_output_list'][0][hand]['pred_hand_pose'].reshape(48)
        wrist_orientation = hand_pose[:3]
        oris.append(wrist_orientation)
    return np.vstack(oris)


if __name__ == '__main__':
    normalize_params_path = 'ori_normalization_params.pkl'
    normalize_mocap_paths = []
    ss_dir = '/scratch/junyao/Datasets/something_something_processed'
    normalize_tasks = [
        'move_away',
        'move_towards',
        'move_down',
        'move_up',
        'pull_left',
        'pull_right',
        'push_left',
        'push_right',
    ]
    normalize_task_dirs = [osp.join(ss_dir, t) for t in normalize_tasks]
    print('Collecting mocap paths.')
    for task_dir in tqdm(normalize_task_dirs, desc='Going through task directories'):
        for split in [
            'train',
            'valid'
        ]:
            split_dir = osp.join(task_dir, split)
            iou_json_path = [osp.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.json')][0]
            with open(iou_json_path, 'r') as f:
                iou_json_dict = json.load(f)
            for vid_num in iou_json_dict:
                normalize_mocap_paths.extend(
                    [osp.join(split_dir, 'mocap_output', vid_num, 'mocap',
                              f'frame{frame_num}_prediction_result.pkl')
                     for frame_num in iou_json_dict[vid_num]]
                )
    print(f'Finished collecting {len(normalize_mocap_paths)} mocap paths.')
    oris = get_all_oris(normalize_mocap_paths)
    normalization_params = {}
    normalization_params['min'] = np.min(oris, axis=0)
    normalization_params['max'] = np.max(oris, axis=0)
    normalization_params['mean'] = np.mean(oris, axis=0)
    normalization_params['std'] = np.std(oris, axis=0)
    for i in range(3):
        plt.hist(oris[:, i], bins=50)
        plt.title(f'orientation {i + 1}')
        plt.savefig(f'ori{i + 1}_hist.png')
        plt.close()

    pprint(normalization_params)
    with open(normalize_params_path, 'wb') as f:
        pickle.dump(normalization_params, f)
