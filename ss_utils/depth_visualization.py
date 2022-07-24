import os
import os.path as osp
import pickle
from pprint import pprint
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def determine_which_hand(hand_info):
    left_hand_exists = len(hand_info['pred_output_list'][0]['left_hand']) > 0
    right_hand_exists = len(hand_info['pred_output_list'][0]['right_hand']) > 0
    if left_hand_exists and not right_hand_exists:
        return 'left_hand'
    if right_hand_exists and not left_hand_exists:
        return 'right_hand'
    if left_hand_exists and right_hand_exists:
        # select the hand with the bigger bounding box
        left_hand_bbox = hand_info['hand_bbox_list'][0]['left_hand']
        *_, lw, lh = left_hand_bbox
        right_hand_bbox = hand_info['hand_bbox_list'][0]['right_hand']
        *_, rw, rh = right_hand_bbox
        if lw * lh >= rw * rh:
            return 'left_hand'
        else:
            return 'right_hand'
    else:
        raise ValueError('No hand detected!')


def normalize_bbox(unnormalized_bbox, img_size):
    img_x, img_y = img_size
    img_x, img_y = float(img_x), float(img_y)

    bbox_0 = unnormalized_bbox[0] / img_x
    bbox_1 = unnormalized_bbox[1] / img_y
    bbox_2 = unnormalized_bbox[2] / img_x
    bbox_3 = unnormalized_bbox[3] / img_y

    return np.array([bbox_0, bbox_1, bbox_2, bbox_3])


def plot_joints(ax, joints_2d_coords):
    for joint_idx, joints_2d_coord in enumerate(joints_2d_coords):
        joint_x, joint_y = joints_2d_coord
        ax.plot(joint_x, joint_y, marker="o", markersize=5, c="r", label=str(joint_idx))
        ax.annotate(str(joint_idx), xy=(joint_x, joint_y), textcoords='data', fontsize=5)


def visualize_frame_joints_depth(vid_dir, vid_num, frame_num):
    frame_path = osp.join(vid_dir, 'frames', str(vid_num), f'frame{frame_num}.jpg')
    depth_path = osp.join(vid_dir, 'depths', str(vid_num), f'frame{frame_num}.npy')
    depth_map_path = osp.join(vid_dir, 'depth_maps', str(vid_num), f'frame{frame_num}.jpg')
    hand_info_path = osp.join(vid_dir, 'mocap_output', str(vid_num), 'mocap', f'frame{frame_num}_prediction_result.pkl')

    # load everything
    frame_bgr = cv2.imread(frame_path)
    depth = np.load(depth_path)
    depth_map_bgr = cv2.imread(depth_map_path)
    with open(hand_info_path, 'rb') as f:
        hand_info = pickle.load(f)

    # get joint coordinates
    hand = determine_which_hand(hand_info)
    joints_3d = hand_info['pred_output_list'][0][hand]['pred_joints_img']
    joints_2d = joints_3d[:, :2]
    joints_2d_coords = joints_2d.round().astype(np.uint64)

    # concat images + highlight joints + legend
    fig, (ax1, ax2) = plt.subplots(1, 2)
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    depth_map = cv2.cvtColor(depth_map_bgr, cv2.COLOR_BGR2RGB)
    ax1.imshow(frame)
    ax2.imshow(depth_map)
    plot_joints(ax1, joints_2d_coords)
    plot_joints(ax2, joints_2d_coords)
    plt.savefig('depth_vis.png', dpi=1200)
    plt.show(dpi=1200)
    plt.close()

    # print everything
    for joint_idx, joints_2d_coord in enumerate(joints_2d_coords):
        joint_x, joint_y = joints_2d_coord
        joint_depth_fm = joints_3d[joint_idx, 2]
        joint_depth_rs = depth[joint_y, joint_x]
        joint_depth_map_val = depth_map[joint_y, joint_x]
        pprint(f'Joint {joint_idx} | '
               f'FrankMocap depth: {joint_depth_fm:.2f} | '
               f'RealSense depth: {joint_depth_rs} | '
               f'RealSense depth map values: {joint_depth_map_val}')
    pprint('Done.')


def visualize_video_wrist_depths(all_vids_dir, vid_dir=None, vid_num=None, depth_descriptor='wrist_img_z'):
    valid_depth_descriptors = ['wrist_img_z', 'bbox_size', 'scaling_factor', 'normalized_bbox_size']
    assert depth_descriptor in valid_depth_descriptors, f'Invalid depth descriptor: {depth_descriptor}.'
    if all_vids_dir is None:
        assert vid_dir is not None and vid_num is not None
        frames_dir = osp.join(vid_dir, 'frames', str(vid_num))
        frame_paths = [osp.join(frames_dir, p) for p in os.listdir(frames_dir)]
        print(f'Visualizing {depth_descriptor} depth '
              f'for {len(frame_paths)} frames in video {vid_num} of {vid_dir}')
    else:
        task_dirs = [osp.join(all_vids_dir, d) for d in os.listdir(all_vids_dir)]
        frame_paths = []
        for task_dir in task_dirs:
            frames_vid_dirs = [osp.join(task_dir, 'frames', d) for d in os.listdir(osp.join(task_dir, 'frames'))]
            for frames_vid_dir in frames_vid_dirs:
                frame_paths.extend([osp.join(frames_vid_dir, p) for p in os.listdir(frames_vid_dir)])
        print(f'Visualizing {depth_descriptor} depth '
              f'for {len(frame_paths)} frames under {all_vids_dir}')

    descriptor_depths, realsense_depths = [], []
    for frame_path in tqdm(frame_paths, desc=f'Going through {len(frame_paths)} frames...'):
        frame_num = int(frame_path.split('/')[-1].split('.')[0][5:])
        vid_num = int(frame_path.split('/')[-2])
        vid_dir = '/' + osp.join(*frame_path.split('/')[:-3])
        depth_path = osp.join(
            vid_dir, 'depths', str(vid_num), f'frame{frame_num}.npy'
        )
        depth = np.load(depth_path)
        ymax, xmax = depth.shape

        hand_info_path = osp.join(
            vid_dir, 'mocap_output', str(vid_num), 'mocap', f'frame{frame_num}_prediction_result.pkl'
        )
        if not osp.exists(hand_info_path):
            print(f'Hand pose in {frame_path} is not detected. Skipping...')
            continue
        with open(hand_info_path, 'rb') as f:
            hand_info = pickle.load(f)

        # get wrist coordinates and realsense depth
        hand = determine_which_hand(hand_info)
        wrist_3d = hand_info['pred_output_list'][0][hand]['pred_joints_img'][0]
        wrist_2d = wrist_3d[:2]
        wrist_x_float, wrist_y_float = wrist_2d
        if not (0 <= wrist_x_float < xmax) or not (0 <= wrist_y_float < ymax):
            print(f'Wrist in {frame_path} is not in the image. Skipping...')
            continue

        wrist_2d_coord = wrist_2d.round().astype(np.uint64)
        wrist_x, wrist_y = wrist_2d_coord
        if wrist_x == xmax or wrist_y == ymax:
            print(f'Wrist in {frame_path} is on image edge. Skipping...')
            continue

        wrist_depth_rs = depth[wrist_y, wrist_x]
        if wrist_depth_rs == 0:
            print(f'Detected 0 RealSense depth for {frame_path}. Skipping...')
            continue

        realsense_depths.append(wrist_depth_rs)

        if depth_descriptor == 'wrist_img_z':
            wrist_depth_fm = wrist_3d[2]
            descriptor_depths.append(wrist_depth_fm)
        elif depth_descriptor == 'bbox_size':
            hand_bbox = hand_info['hand_bbox_list'][0][hand]
            *_, w, h = hand_bbox
            bbox_size = w * h
            descriptor_depths.append(1. / bbox_size)
        elif depth_descriptor == 'scaling_factor':
            cam_scale = hand_info['pred_output_list'][0][hand]['pred_camera'][0]
            hand_boxScale_o2n = hand_info['pred_output_list'][0][hand]['bbox_scale_ratio']
            scaling_factor = cam_scale / hand_boxScale_o2n
            descriptor_depths.append(1. / scaling_factor)
        elif depth_descriptor == 'normalized_bbox_size':
            hand_bbox = hand_info['hand_bbox_list'][0][hand]
            normalized_bbox = normalize_bbox(hand_bbox, (xmax, ymax))
            *_, w, h = normalized_bbox
            normalized_bbox_size = w * h
            descriptor_depths.append(1. / normalized_bbox_size)

    plt.figure()
    descriptor_name = None
    if depth_descriptor == 'wrist_img_z':
        descriptor_name = 'FrankMocap depth'
    elif depth_descriptor == 'bbox_size':
        descriptor_name = 'Bbox size'
    elif depth_descriptor == 'scaling_factor':
        descriptor_name = 'Scaling factor'
    elif depth_descriptor == 'normalized_bbox_size':
        descriptor_name = 'Normalized bbox size'
    plt.title(f'{descriptor_name} vs. RealSense depth')
    plt.scatter(realsense_depths, descriptor_depths, s=1)
    plt.savefig(f'depths_scatter_{depth_descriptor}.png')
    plt.show()
    plt.close()

    print(f'Done. Visualized wrist depths in {len(descriptor_depths)}/{len(frame_paths)} frames.')


def visualize_video_cam_params(vid_dir, vid_num):
    mocap_vid_dir = osp.join(vid_dir, 'mocap_output', str(vid_num), 'mocap')
    hand_info_files = sorted(
        [p for p in os.listdir(mocap_vid_dir)],
        key=lambda x: int(x[5:].split('_')[0])
    )

    cam_scales, cam_xs, cam_ys = [], [], []
    num_frames = len(hand_info_files)
    for hand_info_file in tqdm(hand_info_files, desc=f'Going through {len(hand_info_files)} frames...'):
        hand_info_path = osp.join(mocap_vid_dir, hand_info_file)
        with open(hand_info_path, 'rb') as f:
            hand_info = pickle.load(f)
        hand = determine_which_hand(hand_info)
        cam = hand_info['pred_output_list'][0][hand]['pred_camera']
        cam_scales.append(cam[0])
        cam_xs.append(cam[1])
        cam_ys.append(cam[2])

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    ax1.set_title('Cam scale')
    ax1.plot(range(num_frames), cam_scales)

    ax2.set_title('Cam trans x')
    ax2.plot(range(num_frames), cam_xs)

    ax3.set_title('Cam trans y')
    ax3.plot(range(num_frames), cam_ys)

    plt.tight_layout()
    plt.savefig('cam_vis.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    visualize_frame = False
    visualize_video_depth_wrist_z = False
    visualize_video_depth_bbox = True
    visualize_video_depth_normalized_bbox = True
    visualize_video_depth_scale = False
    visualize_video_cam = False

    all_vids_dir = '/home/junyao/Datasets/something_something_hand_demos'
    vid_dir = '/home/junyao/Datasets/something_something_processed/move_up/valid'
    vid_num = 8746
    frame_num = 14

    if visualize_frame:
        visualize_frame_joints_depth(vid_dir, vid_num, frame_num)

    if visualize_video_depth_wrist_z:
        visualize_video_wrist_depths(all_vids_dir, depth_descriptor='wrist_img_z')

    if visualize_video_depth_bbox:
        visualize_video_wrist_depths(all_vids_dir, depth_descriptor='bbox_size')

    if visualize_video_depth_scale:
        visualize_video_wrist_depths(all_vids_dir, depth_descriptor='scaling_factor')

    if visualize_video_depth_normalized_bbox:
        visualize_video_wrist_depths(all_vids_dir, depth_descriptor='normalized_bbox_size')

    if visualize_video_cam:
        visualize_video_cam_params(vid_dir, vid_num)
