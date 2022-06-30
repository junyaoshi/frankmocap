import json
import os.path
import time
from os.path import join
from tqdm import tqdm
import itertools
import subprocess
import argparse

from ss_utils.ss_dataset import generate_vid_list_from_tasks, convert_videos_to_frames, save_img_shape_to_mocap
from ss_utils.filter_utils import filter_data_by_IoU_threshold

DVD_TASK_TEMPLATES = {
    # "Closing [something]": 0,
    "Moving [something] away from the camera": 1,
    # "Moving [something] towards the camera": 2,
    # "Opening [something]": 3,
    # "Pushing [something] from left to right": 4,
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
ADDITIONAL_TASK_TEMPLATES = {
    # "Poking [something] so it slightly moves": 0,
    # "Picking [something] up": 1,
    # "Putting [something] on a surface": 2,
    # "Holding [something]": 3,
    # "Lifting [something] up completely without letting it drop down": 4,
    # "Moving [something] closer to [something]": 5,
    "Moving [something] away from [something]": 6

}
SPLITS = [
    'train',
    'valid'
]

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Extract 3D hand poses from something-something videos')
    parser.add_argument('--ss_json_dir', dest='ss_json_dir', type=str, required=True,
                        help='direcotry to something-something json files',
                        default='/home/junyao/Datasets/something_something_original')
    parser.add_argument('--ss_vids_dir', dest='ss_vids_dir', type=str, required=True,
                        help='directory of something-something video dataset',
                        default='/home/junyao/Datasets/something_something_original/something_something')
    parser.add_argument('--data_save_dir', dest='data_save_dir', type=str, required=True,
                        help='directory for saving something_something processed data',
                        default='/home/junyao/Datasets/something_something_processed')
    parser.add_argument('--conda_root', dest='conda_root', type=str, required=True,
                        help='root directory of conda',
                        default='/home/junyao/anaconda3')
    parser.add_argument('--task_name', dest='task_name', type=str, required=True,
                        help='name for the tasks being processed',
                        default=None)
    parser.add_argument('--iou_thresh', dest='iou_thresh', type=float, required=True,
                        help='threshold for filtering data with hand and mesh bbox IoU',
                        default=0.7)
    parser.add_argument('--run_on_cv_server', action='store_true',
                        help='if true, run tasks on cv-server; else, run all tasks on cluster')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='if true, then will only process 10 videos for debugging')
    args = parser.parse_args()
    return args


def main(args, task_templates, splits=('train', 'valid')):
    assert splits
    print('Begin video to 3D hand pose conversion.')
    print(f'The specified tasks are: \n{task_templates}')
    print(f'The specified splits are: {splits}')

    train_dict, valid_dict = generate_vid_list_from_tasks(args.ss_json_dir, task_templates)
    dicts = {}
    if 'train' in splits:
        dicts['train'] = train_dict
    if 'valid' in splits:
        dicts['valid'] = valid_dict
    for split, vid_dict in dicts.items():
        # convert webm to jpg
        print(f'Converting videos to frame images for {split} split.')
        video_nums = list(itertools.chain(*list(vid_dict.values())))
        if args.debug:
            video_nums = video_nums[:10]
        video_names = [f'{video_num}.webm' for video_num in video_nums]
        del video_nums
        frames_dir = join(
            args.data_save_dir,
            args.task_name,
            split,
            'frames'
        )
        convert_videos_to_frames(video_names, args.ss_vids_dir, frames_dir)

        # extract bounding boxes from frames and save them to json
        print(f'Converting frame images to bounding boxes for {split} split.')
        bbs_json_dir = join(args.data_save_dir, args.task_name, split, 'bbs_json')
        fm_python_path = join(args.conda_root, 'envs/frankmocap/bin/python')
        bb_command = f"{fm_python_path} demo_ss.py --cuda "
        bb_command += f"--image_parent_dir={frames_dir} "
        bb_command += f"--json_save_dir={bbs_json_dir} "
        cwd = "/home/junyao/LfHV/hand_object_detector"
        p = subprocess.Popen(bb_command, shell=True, cwd=cwd)
        p.communicate()

        # extract 3D hand pose from bounding boxes and save them to pkl
        print(f'Converting bounding boxes to 3D hand poses for {split} split.')
        mocap_output_dir = join(args.data_save_dir, args.task_name, split, 'mocap_output')
        fm_python_path = join(args.conda_root, 'envs/frankmocap/bin/python')
        fm_command = f"xvfb-run -a {fm_python_path} -m demo.demo_handmocap "
        fm_command += f"--input_dir={bbs_json_dir} "
        fm_command += f"--out_parent_dir={mocap_output_dir} "
        fm_command += f"--save_pred_pkl --save_mesh "
        p = subprocess.Popen(fm_command, shell=True)
        p.communicate()

        # save image shapes to mocap output pkl files
        print(f'Saving image shapes to mocap output pkl files for {split} split.')
        save_img_shape_to_mocap(mocap_parent_dir=mocap_output_dir, run_on_cv_server=args.run_on_cv_server)

        # filter data by IoU threshold
        print(f'Filtering data using {args.iou_thresh} IoU threshold for {split} split.')
        iou_json_path = join(args.data_save_dir, args.task_name, split, f'IoU_{args.iou_thresh}.json')
        if os.path.exists(iou_json_path):
            print('IoU json path already exists. Skip filtering data by IoU...')
        else:
            filter_data_by_IoU_threshold(
                data_dir=join(args.data_save_dir, args.task_name, split),
                IoU_thresh=args.iou_thresh,
                json_path=iou_json_path
            )

if __name__ == '__main__':
    start = time.time()
    args = parse_args()
    main(args, task_templates=ADDITIONAL_TASK_TEMPLATES, splits=SPLITS)
    end = time.time()
    print(f'Done. Time elapsed: {end - start} seconds')
