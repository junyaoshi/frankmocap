import json
import os
import os.path as osp
import pickle

from tqdm import tqdm


data_home_dir = '/home/junyao/Datasets/something_something_processed'
has_subdirs = True


if __name__ == '__main__':
    n_fixed, n_good, n_empty, n_total = 0, 0, 0, 0

    if has_subdirs:
        subdirs = [osp.join(data_home_dir, d) for d in os.listdir(data_home_dir)]
    else:
        subdirs = [data_home_dir]
    for subdir in tqdm(subdirs, desc=f'Going through {len(subdirs)} subdirectories'):
        print(f'\nChecking information at: {subdir}')
        mocap_out_dir = osp.join(subdir, 'mocap_output')
        vid_nums = [d for d in os.listdir(mocap_out_dir)]
        for vid_num in vid_nums:
            vid_mocap_dir = osp.join(mocap_out_dir, vid_num, 'mocap')
            frame_nums = sorted([int(p.split('_')[0][5:]) for p in os.listdir(vid_mocap_dir)])
            for frame_num in frame_nums:
                n_total += 1
                frame_mocap_path = osp.join(vid_mocap_dir, f'frame{frame_num}_prediction_result.pkl')
                try:
                    with open(frame_mocap_path, 'rb') as f:
                        frame_mocap = pickle.load(f)
                    img_path = frame_mocap['image_path']
                    if osp.exists(img_path):
                        n_good += 1
                        continue
                    else:
                        fixed_img_path = osp.join(subdir, 'frames', vid_num, f'frame{frame_num}.jpg')
                        frame_mocap['image_path'] = fixed_img_path
                        with open(frame_mocap_path, 'wb') as f:
                            pickle.dump(frame_mocap, f)
                        n_fixed += 1
                except EOFError:
                    n_empty += 1

    print(f'Total: {n_total}; Good: {n_good}; Fixed: {n_fixed}; Empty: {n_empty}.')
    print('Done.')