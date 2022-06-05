import json
import os
from os.path import join
from tqdm import tqdm
import cv2
import multiprocessing as mp
import math
from itertools import repeat

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
        video_frames_dir = join(frames_dir, video_name[:-5])
        os.makedirs(video_frames_dir)
        for idx, frame in frames.items():
            cv2.imwrite(join(video_frames_dir, f'frame{idx}.jpg'), frame)


def convert_videos_to_frames(all_video_names, ss_vids_dir, frames_dir):
    if os.path.exists(frames_dir):
        print(f'Frames directory already exists. Skip video to frame conversion...')
        return

    os.makedirs(frames_dir)
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


if __name__ == '__main__':
    task_templates = {
        # "Closing [something]": 0,
        # "Moving [something] away from the camera": 1,
        # "Moving [something] towards the camera": 2,
        # "Opening [something]": 3,
        "Pushing [something] from left to right": 4,
        "Pushing [something] from right to left": 5,
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
    ss_json_dir = "/home/junyao/Datasets/something_something_original"
    train_dict, valid_dict = generate_vid_list_from_tasks(ss_json_dir, task_templates)
