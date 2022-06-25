import json
import os
from os.path import join
from tqdm import tqdm
import cv2
import multiprocessing as mp
import math
from itertools import repeat
import itertools
import pickle
import numpy as np

# SS_VIDS_PARENT_DIR = '/home/junyao/Datasets/something_something_original'
# SS_VIDS_DIR = join(SS_VIDS_PARENT_DIR, 'something_something')
SS_SAVE_DIR = '/home/junyao/Datasets/something_something_processed'
TASK_TEMPLATES = {
    "Closing [something]": 0,
    "Moving [something] away from the camera": 1,
    "Moving [something] towards the camera": 2,
    "Opening [something]": 3,
    "Pushing [something] from left to right": 4,
    "Pushing [something] from right to left": 5,
    "Poking [something] so lightly that it doesn't or almost doesn't move": 6,
    "Moving [something] down": 7,
    "Moving [something] up": 8,
    "Pulling [something] from left to right": 9,
    "Pulling [something] from right to left": 10,
    "Pushing [something] with [something]": 11,
    "Moving [something] closer to [something]": 12,
    "Plugging [something] into [something]": 13,
    "Pushing [something] so that it slightly moves": 14
}  # these are the tasks used by DVD paper


def generate_vid_list_from_tasks(ss_json_dir, task_templates):
    """
    Args:
        task_templates: a dictionary of task templates, see boave
    Returns:
        train_dict, valid_dict: dictionary containing associated video numbers
    """
    print('Generating video numbers from tasks...')
    # load json
    train_list = json.load(open(join(ss_json_dir, 'something-something-v2-train.json'), 'r'))
    valid_list = json.load(open(join(ss_json_dir, 'something-something-v2-validation.json'), 'r'))

    # split generator
    train_dict, valid_dict = {}, {}
    for k in task_templates.keys():
        train_dict[k] = []
        valid_dict[k] = []
    for train_data in tqdm(train_list, desc='Parsing training json'):
        if train_data['template'] in task_templates.keys():
            train_dict[train_data['template']].append(train_data['id'])
    for valid_data in tqdm(valid_list, desc='Parsing validation json'):
        if valid_data['template'] in task_templates.keys():
            valid_dict[valid_data['template']].append(valid_data['id'])

    # print and visualize
    for k, v in train_dict.items():
        print(f'Train | {k} : {len(v)}')
    for k, v in valid_dict.items():
        print(f'Valid | {k} : {len(v)}')
    train_n_vids = sum([len(v) for v in train_dict.values()])
    print(f'There are {train_n_vids} videos of specified tasks in training set')
    valid_n_vids = sum([len(v) for v in valid_dict.values()])
    print(f'There are {valid_n_vids} videos of specified tasks in validation set')
    print(f'There are {train_n_vids + valid_n_vids} videos of specified tasks in total')

    return train_dict, valid_dict


def single_process_videos_to_frames(video_names, ss_vids_dir, frames_dir):
    for video_name in video_names:
        video_frames_dir = join(frames_dir, video_name[:-5])
        if os.path.exists(video_frames_dir):
            # print(f'skipped {video_frames_dir}')
            continue
        vc = cv2.VideoCapture(join(ss_vids_dir, video_name))
        frames = {}
        frame_idx = 0  # count number of frames
        while True:
            ret, frame = vc.read()
            if ret:
                frames[frame_idx] = frame
                frame_idx += 1
            else:
                break
        vc.release()
        os.makedirs(video_frames_dir)
        for idx, frame in frames.items():
            cv2.imwrite(join(video_frames_dir, f'frame{idx}.jpg'), frame)


def convert_videos_to_frames(all_video_names, ss_vids_dir, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    # all_video_names = tf.io.gfile.listdir(_VIDEO_DIR)
    num_videos = len(all_video_names)
    num_cpus = mp.cpu_count()

    # split into multiple jobs
    splits = list(range(0, num_videos, math.ceil(num_videos / num_cpus)))
    splits.append(num_videos)
    args_list = [all_video_names[splits[i]:splits[i + 1]] for i in range(len(splits) - 1)]

    # # debug
    # single_process_videos_to_frames(args_list[0], frames_dir)

    # multiprocessing (num_cpus processes)
    pool = mp.Pool(num_cpus)
    with pool as p:
        r = list(tqdm(p.starmap(
            single_process_videos_to_frames, zip(args_list, repeat(ss_vids_dir), repeat(frames_dir))
        ), total=num_cpus))


def single_process_save_img_shape_to_mocap(video_mocap_dirs, run_on_cv_server):
    for video_mocap_dir in video_mocap_dirs:
        video_mocap_dir = join(video_mocap_dir, 'mocap')
        frame_mocap_paths = [join(video_mocap_dir, p) for p in os.listdir(video_mocap_dir)]
        for frame_mocap_path in frame_mocap_paths:
            with open(frame_mocap_path, 'rb') as f:
                hand_info = pickle.load(f)
            if 'image_shape' in hand_info:
                continue
            image_path = hand_info['image_path']
            if image_path[:8] == '/scratch' and run_on_cv_server:
                image_path = '/home' + image_path[8:]
            image = cv2.imread(image_path)
            image_shape = np.array(image.shape)[:2]
            hand_info['image_shape'] = image_shape
            with open(frame_mocap_path, 'wb') as f:
                pickle.dump(hand_info, f)


def save_img_shape_to_mocap(mocap_parent_dir, run_on_cv_server):
    all_video_mocap_dirs = [join(mocap_parent_dir, d)for d in os.listdir(mocap_parent_dir)]
    num_videos = len(all_video_mocap_dirs)
    num_cpus = mp.cpu_count()

    splits, n_videos_left, n_cpus_left = [0], num_videos, num_cpus
    while n_videos_left:
        videos_assigned = math.ceil(n_videos_left / n_cpus_left)
        n_videos_left -= videos_assigned
        n_cpus_left -= 1
        last_video = splits[-1]
        splits.append(last_video + videos_assigned)
    args_list = [all_video_mocap_dirs[splits[i]:splits[i + 1]] for i in range(len(splits) - 1)]

    # multiprocessing (num_cpus processes)
    pool = mp.Pool(num_cpus)
    with pool as p:
        r = list(tqdm(p.starmap(
            single_process_save_img_shape_to_mocap, zip(args_list, repeat(run_on_cv_server))
        ), total=num_cpus))


if __name__ == '__main__':
    # test convert_videos_to_frames
    task_templates = {
        # "Closing [something]": 0,
        # "Moving [something] away from the camera": 1,
        # "Moving [something] towards the camera": 2,
        # "Opening [something]": 3,
        "Pushing [something] from left to right": 4,
        # "Pushing [something] from right to left": 5,
        # "Poking [something] so lightly that it doesn't or almost doesn't move": 6,
        # "Moving [something] down": 7,
        # "Moving [something] up": 8,
        # "Pulling [something] from left to right": 9,
        # "Pulling [something] from right to left": 10,
        # "Pushing [something] with [something]": 11,
        # "Moving [something] closer to [something]": 12,
        # "Plugging [something] into [something]": 13,
        # "Pushing [something] so that it slightly moves": 14
    }
    splits = ('valid')
    debug = True
    task_name = 'push_right'

    ss_json_dir = "/home/junyao/Datasets/something_something_original"
    ss_vids_dir = "/home/junyao/Datasets/something_something_original/something_something"
    data_save_dir = "/home/junyao/Datasets/something_something_processed"

    train_dict, valid_dict = generate_vid_list_from_tasks(ss_json_dir, task_templates)
    dicts = {}
    if 'train' in splits:
        dicts['train'] = train_dict
    if 'valid' in splits:
        dicts['valid'] = valid_dict
    for split, vid_dict in dicts.items():
        # convert webm to jpg
        print(f'Converting videos to frame images for {split} split.')
        video_nums = list(itertools.chain(*list(vid_dict.values())))
        if debug:
            video_nums = video_nums[:10]
        video_names = [f'{video_num}.webm' for video_num in video_nums]
        del video_nums
        frames_dir = join(
            data_save_dir,
            task_name,
            split,
            'frames'
        )
        convert_videos_to_frames(video_names, ss_vids_dir, frames_dir)

    # # test single process save image to pkl
    # run_on_cv_server = True
    # video_mocap_dirs = ['/home/junyao/Datasets/something_something_processed/push_right/valid/mocap_output/104468']
    # single_process_save_img_shape_to_mocap(video_mocap_dirs, run_on_cv_server)

    # # test multi process save image to pkl
    # run_on_cv_server = True
    # mocap_parent_dir = '/home/junyao/Datasets/something_something_processed/push_right/valid/mocap_output'
    # save_img_shape_to_mocap(mocap_parent_dir, run_on_cv_server)