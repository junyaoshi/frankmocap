import os
import time
import argparse
import os.path as osp
import subprocess
from tqdm import tqdm
from pprint import pprint

from ss_utils.ss_dataset import save_img_shape_to_mocap
from ss_utils.filter_utils import filter_data_by_IoU_threshold


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Extract 3D hand poses from '
                                                 'demos of something-something tasks')
    parser.add_argument('--demos_dir', type=str,
                        help='directory to something-something hand demo files',
                        default='/home/junyao/Datasets/something_something_hand_demos')
    parser.add_argument('--demo_type', type=str,
                        help='type of demo dataset',
                        default='hand_demos', choices=['hand_demos', 'robot_demos'])
    parser.add_argument('--conda_root', dest='conda_root', type=str,
                        help='root directory of conda',
                        default='/home/junyao/anaconda3')
    parser.add_argument('--iou_thresh', dest='iou_thresh', type=float, required=True,
                        help='threshold for filtering data with hand and mesh bbox IoU',
                        default=0.7)
    args = parser.parse_args()
    return args


def main(args):
    print(f'Begin processing something-something {args.demo_type} dataset.')

    task_dirs = [osp.join(args.demos_dir, d) for d in os.listdir(args.demos_dir)]
    pprint(f'Task directories are: \n{task_dirs}')
    for task_dir in tqdm(task_dirs, desc=f'Processing {len(task_dirs)} task directories...'):
        # extract bounding boxes from frames and save them to h5py
        print(f'\nConverting frames at {osp.join(task_dir, "frames")}'
              f'\nto bounding boxes at {osp.join(task_dir, "bbs_json")}.')
        fm_python_path = osp.join(args.conda_root, 'envs/frankmocap/bin/python')
        bb_command = f"{fm_python_path} demo_ss.py --cuda "
        bb_command += f"--image_parent_dir={osp.join(task_dir, 'frames')} "
        bb_command += f"--json_save_dir={osp.join(task_dir, 'bbs_json')} "
        cwd = "/home/junyao/LfHV/hand_object_detector"
        p = subprocess.Popen(bb_command, shell=True, cwd=cwd)
        p.communicate()

        # extract 3D hand pose from bounding boxes and save them to pkl
        print(f'\nConverting bounding boxes at {osp.join(task_dir, "bbs_json")} '
              f'\nto 3D hand poses at {osp.join(task_dir, "mocap_output")}.')
        fm_command = f"xvfb-run -a {fm_python_path} -m demo.demo_handmocap "
        fm_command += f"--input_dir={osp.join(task_dir, 'bbs_json')} "
        fm_command += f"--out_parent_dir={osp.join(task_dir, 'mocap_output')} "
        fm_command += f"--save_pred_pkl --save_mesh "
        p = subprocess.Popen(fm_command, shell=True)
        p.communicate()

        # save image shapes to mocap output pkl files
        print(f'\nSaving image shapes to mocap output under {osp.join(task_dir, "mocap_output")}.')
        save_img_shape_to_mocap(
            mocap_parent_dir=osp.join(task_dir, 'mocap_output'), run_on_cv_server=True
        )

        # filter data by IoU threshold
        print(f'\nFiltering data using {args.iou_thresh} IoU threshold.')
        iou_json_path = osp.join(task_dir, f'IoU_{args.iou_thresh}.json')
        print(f'Saving IoU json to: {iou_json_path}.')
        if os.path.exists(iou_json_path):
            print('IoU json path already exists. Skip filtering data by IoU...')
        else:
            filter_data_by_IoU_threshold(
                data_dir=osp.join(task_dir),
                IoU_thresh=args.iou_thresh,
                json_path=iou_json_path
            )

        # save r3m embeddings
        r3m_command = f"{fm_python_path} save_r3m_for_ss.py "
        r3m_command += f"--input_dir={task_dir} "
        r3m_command += f"--iou_thresh={args.iou_thresh} "
        cwd = "/home/junyao/LfHV/r3m"
        p = subprocess.Popen(r3m_command, shell=True, cwd=cwd)
        p.communicate()

        if args.demo_type == 'robot_demos':
            r3m_command += f"--robot_demos "
            p = subprocess.Popen(r3m_command, shell=True, cwd=cwd)
            p.communicate()


if __name__ == "__main__":
    t0 = time.time()
    args = parse_args()
    main(args)
    t1 = time.time()
    print(f'Done. Time elapsed: {t1 - t0} seconds')
