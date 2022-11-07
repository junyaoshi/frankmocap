import json
import os
import os.path as osp
import pickle
from pprint import pprint

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ss_utils.filter_utils import determine_which_hand


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


def extract_frame_depth_info(hand_info_path, depth_path, depth_descriptor='scaling_factor'):
    descriptor_depth, realsense_depth = None, None
    nohand, wrist_out, wrist_edge, realsense_0 = False, False, False, False

    depth = np.load(depth_path)
    ymax, xmax = depth.shape

    if not osp.exists(hand_info_path):
        # print(f'Hand pose in {frame_path} is not detected. Skipping...')
        nohand = True
        return (descriptor_depth, realsense_depth), (nohand, wrist_out, wrist_edge, realsense_0)
    with open(hand_info_path, 'rb') as f:
        hand_info = pickle.load(f)

    # get wrist coordinates and realsense depth
    hand = determine_which_hand(hand_info)
    wrist_3d = hand_info['pred_output_list'][0][hand]['pred_joints_img'][0]
    wrist_2d = wrist_3d[:2]
    wrist_x_float, wrist_y_float = wrist_2d
    if not (0 <= wrist_x_float < xmax) or not (0 <= wrist_y_float < ymax):
        # print(f'Wrist in {frame_path} is not in the image. Skipping...')
        wrist_out = True
        return (descriptor_depth, realsense_depth), (nohand, wrist_out, wrist_edge, realsense_0)

    wrist_2d_coord = wrist_2d.round().astype(np.uint64)
    wrist_x, wrist_y = wrist_2d_coord
    if wrist_x == xmax or wrist_y == ymax:
        # print(f'Wrist in {frame_path} is on image edge. Skipping...')
        wrist_edge = True
        return (descriptor_depth, realsense_depth), (nohand, wrist_out, wrist_edge, realsense_0)

    wrist_depth_rs = depth[wrist_y, wrist_x]
    if wrist_depth_rs == 0:
        # print(f'Detected 0 RealSense depth for {frame_path}. Skipping...')
        realsense_0 = True
        return (descriptor_depth, realsense_depth), (nohand, wrist_out, wrist_edge, realsense_0)
    realsense_depth = wrist_depth_rs

    if depth_descriptor == 'wrist_img_z':
        descriptor_depth = wrist_3d[2]
    elif depth_descriptor == 'bbox_size':
        hand_bbox = hand_info['hand_bbox_list'][0][hand]
        *_, w, h = hand_bbox
        bbox_size = w * h
        descriptor_depth = 1. / bbox_size
    elif depth_descriptor == 'scaling_factor':
        cam_scale = hand_info['pred_output_list'][0][hand]['pred_camera'][0]
        hand_boxScale_o2n = hand_info['pred_output_list'][0][hand]['bbox_scale_ratio']
        scaling_factor = cam_scale / hand_boxScale_o2n
        descriptor_depth = 1. / scaling_factor
    elif depth_descriptor == 'normalized_bbox_size':
        hand_bbox = hand_info['hand_bbox_list'][0][hand]
        normalized_bbox = normalize_bbox(hand_bbox, (xmax, ymax))
        *_, w, h = normalized_bbox
        normalized_bbox_size = w * h
        descriptor_depth = 1. / normalized_bbox_size

    return (descriptor_depth, realsense_depth), (nohand, wrist_out, wrist_edge, realsense_0)


def depths_linear_fit(
        frame_paths,
        depth_descriptor='scaling_factor',
        test=False, m=None, b=None
):
    valid_depth_descriptors = ['wrist_img_z', 'bbox_size', 'scaling_factor', 'normalized_bbox_size']
    assert depth_descriptor in valid_depth_descriptors, f'Invalid depth descriptor: {depth_descriptor}.'
    if test:
        assert m is not None and b is not None
        print(f'Testing linear fit using {depth_descriptor} depth for {len(frame_paths)} frames.')
    else:
        print(f'Performing linear fit using {depth_descriptor} depth for {len(frame_paths)} frames.')

    descriptor_depths, realsense_depths = [], []
    n_nohand, n_wrist_out, n_wrist_edge, n_realsense_0 = 0, 0, 0, 0
    for frame_path in tqdm(frame_paths, desc=f'Going through {len(frame_paths)} frames...'):
        frame_num = int(frame_path.split('/')[-1].split('.')[0][5:])
        vid_num = int(frame_path.split('/')[-2])
        vid_dir = '/' + osp.join(*frame_path.split('/')[:-3])
        depth_path = osp.join(
            vid_dir, 'depths', str(vid_num), f'frame{frame_num}.npy'
        )
        hand_info_path = osp.join(
            vid_dir, 'mocap_output', str(vid_num), 'mocap', f'frame{frame_num}_prediction_result.pkl'
        )

        (descriptor_depth, realsense_depth), \
        (nohand, wrist_out, wrist_edge, realsense_0) = extract_frame_depth_info(
            hand_info_path=hand_info_path,
            depth_path=depth_path,
            depth_descriptor=depth_descriptor
        )

        n_nohand += nohand
        n_wrist_out += wrist_out
        n_wrist_edge += wrist_edge
        n_realsense_0 += realsense_0

        if not (nohand or wrist_out or wrist_edge or realsense_0):
            descriptor_depths.append(descriptor_depth)
            realsense_depths.append(realsense_depth)

    print(f'Gone through all frames. '
          f'Total: {len(frame_paths)}; '
          f'Valid: {len(frame_paths) - n_nohand - n_wrist_out - n_wrist_edge - n_realsense_0}')
    print(f'No hand detected: {n_nohand}; '
          f'Wrist out of bound: {n_wrist_out}; '
          f'Wrist on edge: {n_wrist_edge}; '
          f'RealSense 0: {n_realsense_0}.')

    # linear fit
    descriptor_depths, realsense_depths = np.array(descriptor_depths), np.array(realsense_depths)
    if not test:
        m, b = np.polyfit(descriptor_depths, realsense_depths, deg=1)
    yhat = m * descriptor_depths + b
    mse = np.mean((yhat - realsense_depths)**2)
    pearson_corr = np.corrcoef(yhat, realsense_depths)[0, 1]
    print(f'MSE: {mse}; Pearson correlation {pearson_corr}.')

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
    plt.title(f'{"Test" if test else "Train"}: {descriptor_name} vs. RealSense depth')
    plt.scatter(descriptor_depths, realsense_depths, s=0.5)
    plt.plot(descriptor_depths, yhat, label=f'$R^2$: {pearson_corr:.3f}; mse: {mse:.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'depths_scatter_{depth_descriptor}.png')
    plt.show()
    plt.close()

    print(f'Done. Visualized wrist depths in {len(descriptor_depths)}/{len(frame_paths)} frames.')
    return m, b


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


def get_all_depths(mocap_paths, descriptors):
    depths_dict = {d: [] for d in descriptors}
    for mocap_path in tqdm(mocap_paths, desc=f'Going through {len(mocap_paths)} frames.'):
        with open(mocap_path, 'rb') as f:
            hand_info = pickle.load(f)
        hand = determine_which_hand(hand_info)
        img_shape = hand_info['image_shape']
        ymax, xmax = img_shape
        for descriptor in descriptors:
            if descriptor == 'scaling_factor':
                cam_scale = hand_info['pred_output_list'][0][hand]['pred_camera'][0]
                hand_boxScale_o2n = hand_info['pred_output_list'][0][hand]['bbox_scale_ratio']
                scaling_factor = cam_scale / hand_boxScale_o2n
                depth = 1. / scaling_factor
            elif descriptor == 'bbox_size':
                hand_bbox = hand_info['hand_bbox_list'][0][hand]
                *_, w, h = hand_bbox
                bbox_size = w * h
                depth = 1. / bbox_size
            elif descriptor == 'normalized_bbox_size':
                hand_bbox = hand_info['hand_bbox_list'][0][hand]
                normalized_bbox = normalize_bbox(hand_bbox, (xmax, ymax))
                *_, w, h = normalized_bbox
                normalized_bbox_size = w * h
                depth = 1. / normalized_bbox_size
            else:
                raise ValueError(f'Invalid descriptor type: {descriptor}.')
            depths_dict[descriptor].append(depth)
    return depths_dict

if __name__ == '__main__':
    visualize_frame = False
    train_depth_linear_fit = True
    test_depth_linear_fit = False
    save_linear_fit_params = False
    save_depth_normalization_params = False
    visualize_video_cam = False

    descriptors = [
        'scaling_factor',
        'normalized_bbox_size'
    ]

    hand_demos_dir = '/home/junyao/WidowX_Datasets/something_something_hand_demos'
    robot_demos_dir = '/home/junyao/WidowX_Datasets/something_something_robot_demos'
    pre_interaction_dir = '/home/junyao/WidowX_Datasets/something_something_pre_interaction_check_online'
    hand_robot_paired_dir = '/home/junyao/WidowX_Datasets/something_something_hand_robot_paired'
    franka_hand_demos_dir = '/home/junyao/Franka_Datasets/something_something_hand_demos'
    franka_hand_robot_dir = '/home/junyao/Franka_Datasets/something_something_hand_robot_paired'

    vid_dir = '/home/junyao/Datasets/something_something_processed/move_up/valid'
    vid_num = 8746
    frame_num = 14

    if visualize_frame:
        visualize_frame_joints_depth(vid_dir, vid_num, frame_num)

    linear_fit_params = {}
    if train_depth_linear_fit:
        train_frame_paths = []
        train_root_dirs = [
            hand_demos_dir,
            robot_demos_dir,
            hand_robot_paired_dir,
            pre_interaction_dir,
            franka_hand_demos_dir,
            franka_hand_robot_dir
        ]
        train_task_dirs = []
        for data_root_dir in train_root_dirs:
            train_task_dirs.extend([osp.join(data_root_dir, d) for d in os.listdir(data_root_dir)])
        for task_dir in train_task_dirs:
            iou_json_path = [osp.join(task_dir, f) for f in os.listdir(task_dir) if f.endswith('.json')][0]
            with open(iou_json_path, 'r') as f:
                iou_json_dict = json.load(f)
            for vid_num in iou_json_dict:
                train_frame_paths.extend([osp.join(task_dir, 'frames', vid_num, f'frame{frame_num}.jpg')
                                          for frame_num in iou_json_dict[vid_num]])

        test_frame_paths = []
        test_root_dirs = [
            hand_demos_dir,
            # robot_demos_dir,
            # pre_interaction_dir,
            # hand_robot_paired_dir,
            # franka_hand_demos_dir,
            # franka_hand_robot_dir
        ]
        test_task_dirs = []
        for data_root_dir in test_root_dirs:
            test_task_dirs.extend([osp.join(data_root_dir, d) for d in os.listdir(data_root_dir)])
        for task_dir in test_task_dirs:
            iou_json_path = [osp.join(task_dir, f) for f in os.listdir(task_dir) if f.endswith('.json')][0]
            with open(iou_json_path, 'r') as f:
                iou_json_dict = json.load(f)
            for vid_num in iou_json_dict:
                test_frame_paths.extend([osp.join(task_dir, 'frames', vid_num, f'frame{frame_num}.jpg')
                                         for frame_num in iou_json_dict[vid_num]])

        linear_fit_params_path = 'depth_linear_fit_params.pkl'

        for descriptor in descriptors:
            m, b = depths_linear_fit(
                train_frame_paths,
                depth_descriptor=descriptor,
                test=False
            )
            if test_depth_linear_fit:
                depths_linear_fit(
                    test_frame_paths,
                    depth_descriptor=descriptor,
                    test=True,
                    m=m,
                    b=b
                )
            if save_linear_fit_params:
                linear_fit_params[descriptor] = {'m': m, 'b': b}

        if save_linear_fit_params:
            with open(linear_fit_params_path, 'wb') as f:
                pickle.dump(linear_fit_params, f)

    if save_depth_normalization_params:
        normalize_params_path = 'depth_normalization_params.pkl'
        normalize_mocap_paths = []
        cluster_ss_dir = '/scratch/junyao/Datasets/something_something_processed'
        normalize_tasks = [
            'move_away',
            'move_towards',
            'move_down',
            'move_up',
            'pull_left',
            'pull_right',
            'push_left',
            'push_right',
        ]
        normalize_task_dirs = [osp.join(cluster_ss_dir, t) for t in normalize_tasks]
        for task_dir in normalize_task_dirs:
            for split in ['train', 'valid']:
                split_dir = osp.join(task_dir, split)
                iou_json_path = [osp.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.json')][0]
                with open(iou_json_path, 'r') as f:
                    iou_json_dict = json.load(f)
                for vid_num in iou_json_dict:
                    normalize_mocap_paths.extend(
                        [osp.join(split_dir, 'mocap_output', vid_num, 'mocap',
                                  f'frame{frame_num}_prediction_result.pkl')
                         for frame_num in iou_json_dict[vid_num]]
                    )
        depths_dict = get_all_depths(
            mocap_paths=normalize_mocap_paths,
            descriptors=descriptors
        )
        normalization_params = {d: {} for d in descriptors}
        for descriptor in descriptors:
            depths = depths_dict[descriptor]
            normalization_params[descriptor]['min'] = np.min(depths)
            normalization_params[descriptor]['max'] = np.max(depths)
            normalization_params[descriptor]['mean'] = np.mean(depths)
            normalization_params[descriptor]['std'] = np.std(depths)
            plt.hist(depths, bins=50)
            plt.title(f'{descriptor}')
            plt.savefig(f'{descriptor}_hist.png')

        normalization_params['scaling_factor']['scale'] = 2.5
        normalization_params['normalized_bbox_size']['scale'] = 0.01

        with open(normalize_params_path, 'wb') as f:
            pickle.dump(normalization_params, f)

    if visualize_video_cam:
        visualize_video_cam_params(vid_dir, vid_num)
