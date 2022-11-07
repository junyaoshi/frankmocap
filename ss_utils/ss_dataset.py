import json
import os
import os.path as osp
from os.path import join
import multiprocessing as mp
import math
from itertools import repeat
import itertools
import pickle

from tqdm import tqdm
import cv2
import numpy as np
from scipy.signal import savgol_filter

from ss_utils.filter_utils import determine_which_hand

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


def generate_example_data_from_task(task, ss_json_dir):
    # load json
    train_list = json.load(open(join(ss_json_dir, 'something-something-v2-train.json'), 'r'))
    # valid_list = json.load(open(join(ss_json_dir, 'something-something-v2-validation.json'), 'r'))

    # split generator
    data = []
    for train_data in tqdm(train_list, desc='Parsing training json'):
        if train_data['template'] == task:
            data.append(train_data)
        if len(data) == 10:
            break
    return data

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
    print(f'Processing {num_videos} videos with {num_cpus} cpus.')

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


def single_process_save_info_to_mocap(
        video_mocap_dirs,
        run_on_cv_server,
        save_img_shape=True,
        save_contact=True,
):
    n_total, n_error = 0, 0
    n_shape_processed, n_contact_processed = 0, 0
    n_shape_skipped, n_contact_skipped = 0, 0
    for video_mocap_dir in video_mocap_dirs:
        video_num = osp.basename(video_mocap_dir)
        video_bbs_dir = join(osp.dirname(osp.dirname(video_mocap_dir)), 'bbs_json', video_num)
        frame_mocap_paths = [join(video_mocap_dir, 'mocap', p) for p in os.listdir(join(video_mocap_dir, 'mocap'))]
        for frame_mocap_path in frame_mocap_paths:
            n_total += 1
            try:
                with open(frame_mocap_path, 'rb') as f:
                    hand_info = pickle.load(f)
            except EOFError:
                n_error += 1
                # print(f'Encountered empty pickle file: {frame_mocap_path}.')
                continue

            save_info = False
            if save_img_shape:
                if 'image_shape' in hand_info:
                    n_shape_skipped += 1
                else:
                    image_path = hand_info['image_path']
                    if image_path[:8] == '/scratch' and run_on_cv_server:
                        image_path = '/home' + image_path[8:]
                    image = cv2.imread(image_path)
                    image_shape = np.array(image.shape)[:2]
                    hand_info['image_shape'] = image_shape
                    n_shape_processed += 1
                    save_info = True
            if save_contact:
                if 'contact_list' in hand_info:
                    n_contact_skipped += 1
                else:
                    frame_name = osp.basename(frame_mocap_path).split('_')[0]
                    frame_bbs_path = join(video_bbs_dir, f'{frame_name}.json')
                    with open(frame_bbs_path, 'r') as f:
                        hand_json_dict = json.load(f)
                    contact_list = hand_json_dict['contact_list']
                    hand_info['contact_list'] = contact_list
                    n_contact_processed += 1
                    save_info = True
            if save_info:
                with open(frame_mocap_path, 'wb') as f:
                    pickle.dump(hand_info, f)

    return n_total, n_error, n_shape_processed, n_contact_processed, n_shape_skipped, n_contact_skipped


def save_info_to_mocap(mocap_parent_dir, run_on_cv_server,
                       save_img_shape=True, save_contact=True):
    if not (save_img_shape or save_contact):
        print(f'No save flag is set to true. Skipping saving info to mocap at: {mocap_parent_dir}')
        return
    all_video_mocap_dirs = [join(mocap_parent_dir, d)for d in os.listdir(mocap_parent_dir)]
    num_videos = len(all_video_mocap_dirs)
    num_cpus = mp.cpu_count()
    print(f'Processing {num_videos} mocap dirs with {num_cpus} cpus under: \n{mocap_parent_dir}.')

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
            single_process_save_info_to_mocap,
            zip(args_list, repeat(run_on_cv_server), repeat(save_img_shape), repeat(save_contact))
        ), total=num_cpus))

    n_total, n_error = 0, 0
    n_shape_processed, n_contact_processed = 0, 0
    n_shape_skipped, n_contact_skipped = 0, 0
    for info in r:
        n_total += info[0]
        n_error += info[1]
        n_shape_processed += info[2]
        n_contact_processed += info[3]
        n_shape_skipped += info[4]
        n_contact_skipped += info[5]

    print(f'Saved info to mocap. Total frames: {n_total}; Empty frames: {n_error};\n'
          f'Processed image shapes: {n_shape_processed}; Skipped image shapes: {n_shape_skipped};\n'
          f'Processed contact: {n_contact_processed}; Skipped contact: {n_contact_skipped}.')


def single_process_filter_contact_with_savgol(video_mocap_dirs, savgol_params_path):
    n_total, n_error, n_processed, n_skipped = 0, 0, 0, 0
    with open(savgol_params_path, 'rb') as f:
        savgol_params = pickle.load(f)
    window, degree = savgol_params['window_length'], savgol_params['polyorder']

    for video_mocap_dir in video_mocap_dirs:
        mocap_dir = join(video_mocap_dir, 'mocap')
        frame_nums = sorted([int(p.split('_')[0][5:]) for p in os.listdir(mocap_dir)])
        contact_dict = {}
        n_total += len(frame_nums)
        save_info = True

        for frame_num in frame_nums:
            mocap_path = osp.join(mocap_dir, f'frame{frame_num}_prediction_result.pkl')
            try:
                with open(mocap_path, 'rb') as f:
                    hand_info = pickle.load(f)
            except EOFError:
                n_error += 1
                # print(f'Encountered empty pickle file: {mocap_path}.')
                continue
            if 'contact_list' not in hand_info:
                n_error += 1
                continue
            if 'contact_filtered' in hand_info:
                # currently skips the entire video if any frame already has contact_filtered
                n_skipped += len(frame_nums)
                save_info = False
                break

            hand = determine_which_hand(hand_info)
            contact_state = hand_info['contact_list'][0][hand]
            contact = 1 if contact_state == 3 else 0
            contact_dict[frame_num] = {'contact': contact, 'hand_info': hand_info}
            n_processed += 1

        if save_info:
            contacts = [contact_dict[n]['contact'] for n in contact_dict]
            if len(contacts) <= window:
                vid_window = len(contacts) - 1 if len(contacts) % 2 == 0 else len(contacts)
            else:
                vid_window = window
            vid_degree = degree
            while vid_degree >= vid_window:
                vid_degree -= 1

            contacts_filtered = savgol_filter(contacts, window_length=vid_window, polyorder=vid_degree)
            contacts_filtered = (contacts_filtered >= 0.5).astype(np.int64)

            for frame_num, contact_filtered in zip(contact_dict, contacts_filtered):
                mocap_path = osp.join(mocap_dir, f'frame{frame_num}_prediction_result.pkl')
                hand_info = contact_dict[frame_num]['hand_info']
                hand_info['contact_filtered'] = contact_filtered
                with open(mocap_path, 'wb') as f:
                    pickle.dump(hand_info, f)

    return n_total, n_error, n_processed, n_skipped


def filter_contact_with_savgol(mocap_parent_dir, savgol_params_path):
    all_video_mocap_dirs = [join(mocap_parent_dir, d) for d in os.listdir(mocap_parent_dir)]
    num_videos = len(all_video_mocap_dirs)
    num_cpus = mp.cpu_count()
    print(f'Processing {num_videos} mocap dirs with {num_cpus} cpus under: \n{mocap_parent_dir}.')

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
            single_process_filter_contact_with_savgol,
            zip(args_list, repeat(savgol_params_path))
        ), total=num_cpus))

    n_total, n_error, n_processed, n_skipped = 0, 0, 0, 0
    for info in r:
        n_total += info[0]
        n_error += info[1]
        n_processed += info[2]
        n_skipped += info[3]

    print(f'Filtered contact with Sav-Gol. Total frames: {n_total}; Empty frames: {n_error};\n'
          f'Processed frames: {n_processed}; Skipped frames: {n_skipped};\n')


if __name__ == '__main__':
    test_convert_video_to_frames = False
    test_save_info_single_process = False
    test_save_info_multi_process = False
    test_generate_example_data = False
    test_savgol_contact_filter_single_process = False
    test_savgol_contact_filter_multi_process = True

    if test_convert_video_to_frames:
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

    if test_save_info_single_process:
        run_on_cv_server = True
        video_mocap_dirs = ['/home/junyao/Datasets/something_something_processed/push_left/valid/mocap_output/189745']
        single_process_save_info_to_mocap(video_mocap_dirs, run_on_cv_server)

    if test_save_info_multi_process:
        run_on_cv_server = True
        mocap_parent_dir = '/home/junyao/Datasets/something_something_processed/move_down/valid/mocap_output'
        save_info_to_mocap(mocap_parent_dir, run_on_cv_server)

    if test_generate_example_data:
        data = generate_example_data_from_task(
            task="Moving [something] away from [something]",
            ss_json_dir="/home/junyao/Datasets/something_something_original"
        )
        for d in data:
            print(d)

    if test_savgol_contact_filter_single_process:
        video_mocap_dirs = ['/home/junyao/Datasets/something_something_processed/push_left/valid/mocap_output/189745']
        savgol_params_path = 'savgol_params.pkl'
        single_process_filter_contact_with_savgol(video_mocap_dirs, savgol_params_path)

    if test_savgol_contact_filter_multi_process:
        mocap_parent_dir = '/scratch/junyao/Datasets/something_something_processed/pull_left/valid/mocap_output'
        savgol_params_path = '/home/junyao/LfHV/frankmocap/ss_utils/savgol_params.pkl'
        filter_contact_with_savgol(mocap_parent_dir, savgol_params_path)
