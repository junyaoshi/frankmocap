import os
import os.path as osp
import pickle

from tqdm import tqdm

SS_DATA_HOME_DIR = '/scratch/junyao/Datasets/something_something_processed'
TASKS = [
    'move_away',
    'move_towards',
    'move_down',
    'move_up',
    'pull_left',
    'pull_right',
    'push_left',
    'push_right',
]


if __name__ == '__main__':
    print(f'The specified tasks are: {TASKS}')
    task_dirs = [osp.join(SS_DATA_HOME_DIR, task) for task in TASKS]
    for task_dir in task_dirs:
        print(f'Checking contact information at: {task_dir}')
        frame_mocap_paths = []
        split_dirs = [osp.join(task_dir, d) for d in os.listdir(task_dir)]
        for split_dir in split_dirs:
            split_mocap_out_dir = osp.join(split_dir, 'mocap_output')
            vid_mocap_dirs = [osp.join(split_mocap_out_dir, d, 'mocap') for d in os.listdir(split_mocap_out_dir)]
            for vid_mocap_dir in vid_mocap_dirs:
                frame_mocap_paths.extend([osp.join(vid_mocap_dir, p) for p in os.listdir(vid_mocap_dir)])

        n_has_contact, n_has_savgol, n_empty, n_bbox_int = 0, 0, 0, 0
        for frame_mocap_path in tqdm(frame_mocap_paths, desc=f'Going through {len(frame_mocap_paths)} frames'):
            try:
                with open(frame_mocap_path, 'rb') as f:
                    frame_mocap = pickle.load(f)
                n_has_contact += 'contact_list' in frame_mocap
                n_has_savgol += 'contact_filtered' in frame_mocap
                left_bbox = frame_mocap['hand_bbox_list'][0]['left_hand']
                right_bbox = frame_mocap['hand_bbox_list'][0]['right_hand']
                if isinstance(left_bbox, int) and isinstance(right_bbox, int):
                    n_bbox_int += 1
            except EOFError:
                n_empty += 1
        print(f'{n_has_contact}/{len(frame_mocap_paths)} frames have contact info.\n'
              f'{n_empty}/{len(frame_mocap_paths)} frames are empty.\n'
              f'{n_has_savgol}/{len(frame_mocap_paths)} frames have Sav-Gol filtered contact info.\n'
              f'{n_bbox_int}/{len(frame_mocap_paths)} frames have bbox bug.\n')
    print('Done.')
