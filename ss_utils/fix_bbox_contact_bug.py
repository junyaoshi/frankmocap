import json
import os
from os.path import join, basename, dirname
import pickle

from tqdm import tqdm

SS_DATA_HOME_DIR = '/scratch/junyao/Datasets/something_something_processed'


if __name__ == '__main__':
    task_dirs = [join(SS_DATA_HOME_DIR, d) for d in os.listdir(SS_DATA_HOME_DIR)]
    for task_dir in task_dirs:
        print(f'Checking information at: {task_dir}')
        frame_mocap_paths = []
        split_dirs = [join(task_dir, d) for d in os.listdir(task_dir)]
        for split_dir in split_dirs:
            split_mocap_out_dir = join(split_dir, 'mocap_output')
            vid_mocap_dirs = [join(split_mocap_out_dir, d, 'mocap') for d in os.listdir(split_mocap_out_dir)]
            for vid_mocap_dir in vid_mocap_dirs:
                frame_mocap_paths.extend([join(vid_mocap_dir, p) for p in os.listdir(vid_mocap_dir)])

        n_bbox_int, n_empty = 0, 0
        for frame_mocap_path in tqdm(frame_mocap_paths, desc=f'Going through {len(frame_mocap_paths)} frames'):
            frame_num = basename(frame_mocap_path).split('_')[0][5:]
            vid_mocap_dir = dirname(dirname(frame_mocap_path))
            vid_num = basename(vid_mocap_dir)
            split_name = basename(dirname(dirname(vid_mocap_dir)))
            frame_json_path = join(task_dir, split_name, 'bbs_json', vid_num, f'frame{frame_num}.json')
            try:
                with open(frame_mocap_path, 'rb') as f:
                    frame_mocap = pickle.load(f)
                left_bbox = frame_mocap['hand_bbox_list'][0]['left_hand']
                right_bbox = frame_mocap['hand_bbox_list'][0]['right_hand']
                if isinstance(left_bbox, int) and isinstance(right_bbox, int):
                    n_bbox_int += 1
                    print(f'\nBug at: {frame_mocap_path}')
                    with open(frame_json_path, 'r') as f:
                        frame_json = json.load(f)
                    frame_mocap['hand_bbox_list'] = frame_json['hand_bbox_list']
                    with open(frame_mocap_path, 'wb') as f:
                        pickle.dump(frame_mocap, f)
            except EOFError:
                n_empty += 1
        print(f'{n_bbox_int}/{len(frame_mocap_paths)} frames have bbox bug.\n'
              f'{n_empty}/{len(frame_mocap_paths)} frames are empty.\n')
    print('Done.')