import os
import os.path as osp
import shutil
from tqdm import tqdm


hand_demo_dir = '/home/junyao/Datasets/something_something_hand_demos'
task_dirs = [osp.join(hand_demo_dir, d) for d in os.listdir(hand_demo_dir)]


for task_dir in tqdm(task_dirs, desc=f'Going through {len(task_dirs)} task directories...'):
    frames_dir = osp.join(task_dir, 'frames')
    depths_dir = osp.join(task_dir, 'depths')
    depth_maps_dir = osp.join(task_dir, 'depth_maps')
    bbs_h5_dir = osp.join(task_dir, 'bbs_h5')
    os.makedirs(bbs_h5_dir, exist_ok=True)

    vid_nums = os.listdir(task_dir)
    for vid_num in vid_nums:
        if not vid_num.isnumeric():
            continue
        src_vid_frames_dir = osp.join(task_dir, vid_num, 'rgb')
        dst_vid_frames_dir = osp.join(frames_dir, vid_num)
        if osp.exists(dst_vid_frames_dir):
            print(f'\nDestination directory {dst_vid_frames_dir} already exists. Directory moving skipped.')
        else:
            shutil.move(src_vid_frames_dir, dst_vid_frames_dir)

        src_vid_depths_dir = osp.join(task_dir, vid_num, 'depth')
        dst_vid_depths_dir = osp.join(depths_dir, vid_num)
        if osp.exists(dst_vid_depths_dir):
            print(f'Destination directory {dst_vid_depths_dir} already exists. Directory moving skipped.')
        else:
            shutil.move(src_vid_depths_dir, dst_vid_depths_dir)

        src_vid_depth_maps_dir = osp.join(task_dir, vid_num, 'depth_map')
        dst_vid_depth_maps_dir = osp.join(depth_maps_dir, vid_num)
        if osp.exists(dst_vid_depth_maps_dir):
            print(f'Destination directory {dst_vid_depth_maps_dir} already exists. Directory moving skipped.')
        else:
            shutil.move(src_vid_depth_maps_dir, dst_vid_depth_maps_dir)

        src_bbs_path = osp.join(task_dir, vid_num, 'bboxes.h5')
        dst_bbs_path = osp.join(bbs_h5_dir, f'{vid_num}.h5')
        if osp.exists(dst_bbs_path):
            print(f'Destination path {dst_bbs_path} already exists. File moving skipped.')
        else:
            shutil.move(src_bbs_path, dst_bbs_path)

        os.rmdir(osp.join(task_dir, vid_num))




