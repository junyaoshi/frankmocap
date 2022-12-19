"""
Visualize number of pre-interaction and during-interaction data
by task, split, and in total.
Print out video and frame statistics.
"""
import os
import os.path as osp
import pickle
import json

from tqdm import tqdm

from ss_utils.filter_utils import determine_which_hand

SS_DATA_HOME_DIR = '/home/junyao/Datasets/something_something_processed'
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
SPLITS = ['train', 'valid']
IOU_THRESH = 0.7
TIME_INTERVAL = 10


def check_contact_sequence(contact_sequence):
    """Check if contact sequence is valid"""
    valid = True
    contact_keys, contact_vals = list(contact_sequence.keys()), list(contact_sequence.values())
    first_contact_frame, last_contact_frame = -1, -1
    for first_contact_idx, frame in enumerate(contact_keys):
        if contact_sequence[frame] == 1:
            first_contact_frame = frame
            break
    for last_contact_idx, frame in reversed(list(enumerate(contact_keys))):
        if contact_sequence[frame] == 1:
            last_contact_frame = frame
            break
    if first_contact_frame == -1 or last_contact_frame == -1:
        valid = False
    else:
        assert first_contact_frame <= last_contact_frame
        for idx in range(first_contact_idx, last_contact_idx):
            if contact_vals[idx] == 0:
                valid = False

    return valid, first_contact_frame, last_contact_frame


def count_frame_pairs_by_stage(frame_pairs, first_contact_frame, last_contact_frame):
    # count number of pre-interaction and during-interaction frame pairs
    pre_count, during_count = 0, 0
    for frame_pair in frame_pairs:
        cur_frame = int(frame_pair[0])
        if cur_frame < first_contact_frame:
            pre_count += 1
        elif first_contact_frame <= cur_frame <= last_contact_frame:
            during_count += 1

    return pre_count, during_count


if __name__ == '__main__':
    print(f'Performing data stage analysis for time interval: {TIME_INTERVAL}.\n')
    all_total, all_pre, all_during, all_invalid, all_vid_total, all_vid_invalid = 0, 0, 0, 0, 0, 0
    split_dict = {
        split: {'total': 0, 'pre': 0, 'during': 0, 'invalid': 0, 'vid_total': 0, 'vid_invalid': 0} for split in SPLITS
    }
    task_dict = {
        task: {'total': 0, 'pre': 0, 'during': 0, 'invalid': 0, 'vid_total': 0, 'vid_invalid': 0} for task in TASKS
    }

    for split in SPLITS:
        for task in TASKS:
            task_dir = osp.join(SS_DATA_HOME_DIR, task)
            frame_mocap_paths = []
            split_dir = osp.join(task_dir, split)
            iou_json_path = osp.join(split_dir, f'IoU_{IOU_THRESH}.json')
            with open(iou_json_path, 'r') as f:
                json_dict = json.load(f)

            total, pre, during, invalid, vid_total, vid_invalid = 0, 0, 0, 0, 0, 0
            # iterate through all videos
            for vid_num in tqdm(json_dict, desc=f'Processing videos in split {split} of task {task}'):
                frame_pairs, contact_sequence = [], {}
                mocap_vid_dir = osp.join(split_dir, 'mocap_output', vid_num, 'mocap')
                vid_total += 1

                # iterate through all valid frames to 1. get contact sequence 2. check all valid (cur, next) frame pairs
                for cur_frame in json_dict[vid_num]:
                    cur_mocap_path = osp.join(mocap_vid_dir, f'frame{cur_frame}_prediction_result.pkl')
                    with open(cur_mocap_path, 'rb') as f:
                        cur_hand_info = pickle.load(f)
                    cur_contact = cur_hand_info['contact_filtered']
                    contact_sequence[int(cur_frame)] = cur_contact

                    # check if future frame exists
                    next_frame = str(int(cur_frame) + TIME_INTERVAL)
                    if next_frame not in json_dict[vid_num]:
                        continue

                    # check if current and future frames have the same hand
                    cur_hand = determine_which_hand(cur_hand_info)

                    next_mocap_path = osp.join(mocap_vid_dir, f'frame{next_frame}_prediction_result.pkl')
                    with open(next_mocap_path, 'rb') as f:
                        next_hand_info = pickle.load(f)
                    next_hand = determine_which_hand(next_hand_info)
                    if cur_hand != next_hand:
                        continue

                    frame_pairs.append((cur_frame, next_frame))

                total += len(frame_pairs)
                contact_sequence = dict(sorted(contact_sequence.items()))
                valid, first_contact_frame, last_contact_frame = check_contact_sequence(contact_sequence)
                if not valid:
                    # print(f'Invalid contact sequence: {contact_sequence}')
                    invalid += len(frame_pairs)
                    vid_invalid += 1
                    continue

                pre_count, during_count = count_frame_pairs_by_stage(
                    frame_pairs, first_contact_frame, last_contact_frame
                )
                pre += pre_count
                during += during_count

            print(f'Frame Stats | Total: {total}; '
                  f'Pre: {pre} ({pre / total * 100:.2f}%); '
                  f'During: {during} ({during / total * 100:.2f}%); '
                  f'Invalid: {invalid} ({invalid / total * 100:.2f}%).')
            print(f'Video Stats | Total: {vid_total}; '
                  f'Invalid: {vid_invalid} ({vid_invalid / vid_total * 100:.2f}%).')

            all_total += total
            all_pre += pre
            all_during += during
            all_invalid += invalid
            all_vid_total += vid_total
            all_vid_invalid += vid_invalid

            split_dict[split]['total'] += total
            split_dict[split]['pre'] += pre
            split_dict[split]['during'] += during
            split_dict[split]['invalid'] += invalid
            split_dict[split]['vid_total'] += vid_total
            split_dict[split]['vid_invalid'] += vid_invalid

            task_dict[task]['total'] += total
            task_dict[task]['pre'] += pre
            task_dict[task]['during'] += during
            task_dict[task]['invalid'] += invalid
            task_dict[task]['vid_total'] += vid_total
            task_dict[task]['vid_invalid'] += vid_invalid

    print(f'\nTotal Summary:')
    print(f'Frame Stats | Total: {all_total}; '
          f'Pre: {all_pre} ({all_pre / all_total * 100:.2f}%); '
          f'During: {all_during} ({all_during / all_total * 100:.2f}%); '
          f'Invalid: {all_invalid} ({all_invalid / all_total * 100:.2f}%).')
    print(f'Video Stats | Total: {all_vid_total}; '
          f'Invalid: {all_vid_invalid} ({all_vid_invalid / all_vid_total * 100:.2f}%).')

    print(f'\nSplit Summary')
    for split, split_stats in split_dict.items():
        print(f"Split {split} Frame Stats | Total: {split_stats['total']}; "
              f"Pre: {split_stats['pre']} ({split_stats['pre'] / split_stats['total'] * 100:.2f}%); "
              f"During: {split_stats['during']} ({split_stats['during'] / split_stats['total'] * 100:.2f}%); "
              f"Invalid: {split_stats['invalid']} ({split_stats['invalid'] / split_stats['total'] * 100:.2f}%).")
        print(f"Split {split} Video Stats | Total: {split_stats['vid_total']}; "
              f"Invalid: {split_stats['vid_invalid']} "
              f"({split_stats['vid_invalid'] / split_stats['vid_total'] * 100:.2f}%).")

    print(f'\nTask Summary')
    for task, task_stats in task_dict.items():
        print(f"Task {task} Frame Stats | Total: {task_stats['total']}; "
              f"Pre: {task_stats['pre']} ({task_stats['pre'] / task_stats['total'] * 100:.2f}%); "
              f"During: {task_stats['during']} ({task_stats['during'] / task_stats['total'] * 100:.2f}%); "
              f"Invalid: {task_stats['invalid']} ({task_stats['invalid'] / task_stats['total'] * 100:.2f}%).")
        print(f"Task {task} Video Stats | Total: {task_stats['vid_total']}; "
              f"Invalid: {task_stats['vid_invalid']} "
              f"({task_stats['vid_invalid'] / task_stats['vid_total'] * 100:.2f}%).")
