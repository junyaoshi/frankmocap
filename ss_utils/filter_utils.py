import json
import os
import time
from os.path import join
import pickle
import shutil

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as im
import cv2
from torchvision.ops import box_iou
import torch

import mocap_utils.demo_utils as demo_utils
from renderer.image_utils import draw_hand_bbox

DATA_DIR = '/home/junyao/Datasets/something_something_processed/push_left_right'


def determine_which_hand(hand_info):
    left_hand_exists = len(hand_info['pred_output_list'][0]['left_hand']) > 0
    right_hand_exists = len(hand_info['pred_output_list'][0]['right_hand']) > 0
    if left_hand_exists and not right_hand_exists:
        return 'left_hand'
    if right_hand_exists and not left_hand_exists:
        return 'right_hand'
    if left_hand_exists and right_hand_exists:
        try:
            # select the hand with the bigger bounding box
            left_hand_bbox = hand_info['hand_bbox_list'][0]['left_hand']
            *_, lw, lh = left_hand_bbox
            right_hand_bbox = hand_info['hand_bbox_list'][0]['right_hand']
            *_, rw, rh = right_hand_bbox
        except TypeError as e:
            print(f'image path: {hand_info["image_path"]}')
            print(f'hand_bbox_list: {hand_info["hand_bbox_list"]}')
            raise e
        if lw * lh >= rw * rh:
            return 'left_hand'
        else:
            return 'right_hand'
    else:
        raise ValueError('No hand detected!')


def get_3d_hand_pose(hand_info):
    """Get 3D hand pose of a frame"""
    hand = hand_info['pred_output_list'][0]['left_hand'] if len(hand_info['pred_output_list'][0]['left_hand']) > 0 \
        else hand_info['pred_output_list'][0]['right_hand']
    joints_smpl = hand['pred_joints_smpl'].reshape(63)
    joints_position = hand['pred_joints_img'].reshape(63)
    joints_angle = hand['pred_hand_pose'].reshape(48)
    return joints_smpl, joints_position, joints_angle


def get_hand_pose_variance(vid_dir):
    """Returns the variance of detected 3D hand pose given a video
    We want to filter out videos that have high variance, which indicates
    Frank Mocap is not able to detect consistent hand poses throughout the video
    """
    vid_hand_pose_path = join(vid_dir, 'mocap')
    frame_hand_pose_paths = [p for p in os.listdir(vid_hand_pose_path) if p.endswith(".pkl")]
    vid_hand_poses = []
    for frame_hand_pose_path in frame_hand_pose_paths:
        frame_hand_info = pickle.load(open(join(vid_hand_pose_path, frame_hand_pose_path), 'rb'))
        frame_hand_pose = get_3d_hand_pose(frame_hand_info)
        vid_hand_poses.append(frame_hand_pose)

    vid_hand_poses = np.array(vid_hand_poses)
    hand_pose_variance = np.sum(np.var(vid_hand_poses, axis=0))

    return hand_pose_variance


def plot_delta_hand_pose(vid_joints_smpl, vid_joints_position, vid_joints_angle,
                         n_frames, vid_dir, plot_dir, frame_pkl_name):
    vid_joints_smpl = np.array(vid_joints_smpl)
    vid_joints_position = np.array(vid_joints_position)
    vid_joints_angle = np.array(vid_joints_angle)

    """Plot delta joint smpl, position, angle vs. time along with original frame"""
    delta_joints_smpl = np.linalg.norm(vid_joints_smpl[1:] - vid_joints_smpl[:-1], axis=1)
    delta_joints_position = np.linalg.norm(vid_joints_position[1:] - vid_joints_position[:-1], axis=1)
    delta_joints_angle = np.linalg.norm(vid_joints_angle[1:] - vid_joints_angle[:-1], axis=1)
    t = len(delta_joints_smpl)
    n_frame = frame_pkl_name.split('_')[0][5:]
    plot_path = join(plot_dir, f'plot{n_frame}.png')

    fig, axs = plt.subplots(4, figsize=(10, 7))
    plt.xlabel('Time')

    # joint smpl
    axs[0].plot(range(1, t + 1), delta_joints_smpl)
    axs[0].set_ylabel('delta smpl')
    axs[0].set_ylim(0, 10)

    # joint position
    axs[1].plot(range(1, t + 1), delta_joints_position)
    axs[1].set_ylabel('delta pos')
    axs[1].set_ylim(0, 400)

    # joint angle
    axs[2].plot(range(1, t + 1), delta_joints_angle)
    axs[2].set_ylabel('delta angle')
    axs[2].set_ylim(0, 20)

    # original image
    im_path = join(vid_dir, 'rendered', f'frame{n_frame}.jpg')
    image = im.imread(im_path)
    axs[3].imshow(image)
    axs[3].axis('off')

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)


def load_hand_info_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        hand_info = pickle.load(f)
    return hand_info


def load_hand_info_from_json(json_path):
    with open(json_path, 'r') as f:
        hand_info = json.load(f)
    return hand_info


def generate_delta_hand_pose_vid(vid_num):
    ss_vid_dir = join(DATA_DIR, str(vid_num))
    vid_mocap_dir = join(ss_vid_dir, 'mocap')
    frame_pkl_names = [n for n in os.listdir(vid_mocap_dir) if n.endswith(".pkl")]
    frame_pkl_names = sorted(frame_pkl_names, key=lambda x: int(x[5:].split('_')[0]))

    vid_joints_smpl, vid_joints_position, vid_joints_angle = [], [], []
    n_frames = len(frame_pkl_names)
    plot_dir = join(ss_vid_dir, 'delta_hand_pose_plots')
    os.makedirs(plot_dir, exist_ok=True)

    for frame_pkl_name in tqdm(frame_pkl_names, desc='Going through frames...'):
        frame_pkl_path = join(vid_mocap_dir, frame_pkl_name)
        frame_hand_info = load_hand_info_from_pkl(frame_pkl_path)
        joints_smpl, joints_position, joints_angle = get_3d_hand_pose(frame_hand_info)

        vid_joints_smpl.append(joints_smpl)
        vid_joints_position.append(joints_position)
        vid_joints_angle.append(joints_angle)

        if len(vid_joints_smpl) == 1:
            continue
        plot_delta_hand_pose(
            vid_joints_smpl, vid_joints_position, vid_joints_angle,
            n_frames, ss_vid_dir, plot_dir, frame_pkl_name
        )

    save_vid_dir = join(ss_vid_dir, 'delta_hand_pose_vid')
    os.makedirs(save_vid_dir, exist_ok=True)
    save_vid_name = f'delta_hand_pose_{vid_num}.mp4'

    plot_names = [img for img in os.listdir(plot_dir) if img.endswith(".png")]
    plot_names = sorted(plot_names, key=lambda x: int(x[4:].split('.')[0]))
    frame = cv2.imread(join(plot_dir, plot_names[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 6
    video = cv2.VideoWriter(join(save_vid_dir, save_vid_name), fourcc, fps, (width, height))

    for plot_name in tqdm(plot_names, desc='Going through saved plots...'):
        video.write(cv2.imread(join(plot_dir, plot_name)))

    cv2.destroyAllWindows()
    video.release()


def extract_mesh_bbox_list(hand_info):
    pred_output_list = hand_info['pred_output_list']

    # extract mesh for rendering (vertices in image space and faces) from pred_output_list
    mesh_bbox_dict = {}
    for hand_type, output in pred_output_list[0].items():
        if output:
            # bbox for verts
            verts = pred_output_list[0][hand_type]['pred_vertices_img']
            x0 = int(np.min(verts[:, 0]))
            x1 = int(np.max(verts[:, 0]))
            y0 = int(np.min(verts[:, 1]))
            y1 = int(np.max(verts[:, 1]))
            w = x1 - x0
            h = y1 - y0
            mesh_bbox_dict[hand_type] = np.array([x0, y0, w, h])
        else:
            mesh_bbox_dict[hand_type] = None

    return [mesh_bbox_dict]


def xywh_2_xyxy(xywh):
    x0, y0, w, h = xywh
    x1, y1 = x0 + w, y0 + h
    return np.array([x0, y0, x1, y1])


def extract_pred_mesh_list(hand_info):
    pred_output_list = hand_info['pred_output_list']
    for k, v in pred_output_list[0].items():
        if not v:
            pred_output_list[0][k] = None

    pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)
    return pred_mesh_list


def calculate_box_iou(hand_bbox_list, mesh_bbox_list):
    ious = []
    for hand_bboxes, mesh_bboxes in zip(hand_bbox_list, mesh_bbox_list):
        for hand_type, hand_bbox in hand_bboxes.items():
            if hand_bbox is not None:
                hand_bbox = torch.Tensor(xywh_2_xyxy(hand_bbox)).unsqueeze(0)
                mesh_bbox = torch.Tensor(xywh_2_xyxy(mesh_bboxes[hand_type])).unsqueeze(0)
                iou = box_iou(hand_bbox, mesh_bbox)
                ious.append(iou.item())

    return np.mean(ious)


def visualize_frame_IoU(frame_pkl_path, res_img_path, visualizer):
    """Visualize the IoU of hand and mesh bboxes; Save result image"""
    hand_info = load_hand_info_from_pkl(frame_pkl_path)
    hand_bbox_list = hand_info['hand_bbox_list']
    mesh_bbox_list = extract_mesh_bbox_list(hand_info)
    img_original_bgr = cv2.imread(hand_info['image_path'])
    pred_mesh_list = extract_pred_mesh_list(hand_info)
    IoU = calculate_box_iou(hand_bbox_list, mesh_bbox_list)

    res_img = img_original_bgr.copy()
    res_img = draw_hand_bbox(res_img, hand_bbox_list)
    res_img = draw_hand_bbox(res_img, mesh_bbox_list)
    res_img = visualizer.render_pred_verts(res_img, pred_mesh_list)

    white = np.zeros((50, res_img.shape[1], 3), np.uint8)
    white[:] = (255, 255, 255)
    res_img = cv2.vconcat((white, res_img))
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(res_img, f'IoU: {IoU:.3f}', (30, 40), font, 0.8, (0, 0, 0), 2, 0)

    cv2.imwrite(res_img_path, res_img)
    print(f"Visualization saved: {res_img_path}")


def visualize_frame_contact(frame_pkl_path, frame_json_path, res_img_path, visualizer):
    """Visualize the contact variable; Save result image"""
    hand_info_pkl = load_hand_info_from_pkl(frame_pkl_path)
    hand_bbox_list = hand_info_pkl['hand_bbox_list']
    image_path = hand_info_pkl['image_path']
    if image_path[:8] == '/scratch':
        image_path = '/home' + image_path[8:]
    img_original_bgr = cv2.imread(image_path)
    pred_mesh_list = extract_pred_mesh_list(hand_info_pkl)

    hand_info_json = load_hand_info_from_json(frame_json_path)
    contact_list = hand_info_json['contact_list']
    if contact_list[0]['left_hand'] != -1:
        contact_state = contact_list[0]['left_hand']
    else:
        contact_state = contact_list[0]['right_hand']
    state_map = {0: 'No Contact', 1: 'Self Contact', 2: 'Another Person', 3: 'Portable Object', 4: 'Stationary Object'}
    contact_str = state_map[contact_state]

    res_img = img_original_bgr.copy()
    res_img = draw_hand_bbox(res_img, hand_bbox_list)
    res_img = visualizer.render_pred_verts(res_img, pred_mesh_list)

    white = np.zeros((50, res_img.shape[1], 3), np.uint8)
    white[:] = (255, 255, 255)
    res_img = cv2.vconcat((white, res_img))
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(res_img, f'Contact: {contact_str}', (30, 40), font, 0.8, (0, 0, 0), 2, 0)

    cv2.imwrite(res_img_path, res_img)
    # print(f"Contact: {contact_str}\nVisualization saved: {res_img_path}")


def generate_frame_IoU_histogram(histogram_path):
    IoUs = []
    train_mocap_output_dir = join(DATA_DIR, 'train', 'mocap_output')
    train_vid_dirs = [join(train_mocap_output_dir, d) for d in os.listdir(train_mocap_output_dir)]
    valid_mocap_output_dir = join(DATA_DIR, 'valid', 'mocap_output')
    valid_vid_dirs = [join(valid_mocap_output_dir, d) for d in os.listdir(valid_mocap_output_dir)]
    vid_dirs = train_vid_dirs + valid_vid_dirs
    for vid_dir in tqdm(
            vid_dirs,
            desc='Going through video directories to generate IoU threshold...'
    ):
        vid_mocap_dir = join(vid_dir, 'mocap')
        frame_pkl_paths = [join(vid_mocap_dir, d) for d in os.listdir(vid_mocap_dir)]
        for frame_pkl_path in frame_pkl_paths:
            hand_info = load_hand_info_from_pkl(frame_pkl_path)
            hand_bbox_list = hand_info['hand_bbox_list']
            mesh_bbox_list = extract_mesh_bbox_list(hand_info)
            IoU = calculate_box_iou(hand_bbox_list, mesh_bbox_list)
            IoUs.append(IoU)

    # plot histogram
    plt.hist(IoUs, bins=20)
    plt.savefig(histogram_path)
    plt.show()
    plt.close()


def filter_data_by_IoU_threshold(data_dir, IoU_thresh, json_path):
    print(f'Processing {data_dir} by IoU threshold {IoU_thresh}.')

    json_dict = {}
    num_valid_data = 0
    num_data = 0
    mocap_output_dir = join(data_dir, 'mocap_output')
    vid_nums = [d for d in os.listdir(mocap_output_dir)]
    for vid_num in tqdm(
            vid_nums,
            desc='Going through video directories to filter data by IoU threshold...'
    ):
        vid_dir = join(mocap_output_dir, vid_num)
        vid_mocap_dir = join(vid_dir, 'mocap')
        frame_pkl_fnames = [d for d in os.listdir(vid_mocap_dir)]
        for frame_pkl_fname in frame_pkl_fnames:
            num_data += 1
            frame_pkl_path = join(vid_mocap_dir, frame_pkl_fname)
            hand_info = load_hand_info_from_pkl(frame_pkl_path)
            hand_bbox_list = hand_info['hand_bbox_list']
            mesh_bbox_list = extract_mesh_bbox_list(hand_info)
            IoU = calculate_box_iou(hand_bbox_list, mesh_bbox_list)
            if IoU >= IoU_thresh:
                num_valid_data += 1
                frame_num = frame_pkl_fname.split('_')[0][5:]
                if vid_num in json_dict:
                    json_dict[vid_num].append(frame_num)
                else:
                    json_dict[vid_num] = [frame_num]

    print(f'There are {num_valid_data} valid frames out of {num_data} frames.')
    print(f'Success rate is {num_valid_data / num_data :.4f}.')
    with open(json_path, 'w') as f:
        json.dump(json_dict, f)


def filter_singe_data_by_IoU_threshold(
        mocap_path,
        IoU_thresh,
        json_path,
        vid_num=None,
        frame_num=None,
        mocap_pred=None,
        verbose=True
):
    if verbose:
        print(f'Processing {mocap_path if mocap_pred is None else "given mocap predictions"} '
              f'by IoU threshold {IoU_thresh}.')

    json_dict = None
    if json_path is not None:
        assert vid_num is not None and frame_num is not None
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                json_dict = json.load(f)
        else:
            json_dict = {}

    assert mocap_pred is not None or mocap_path is not None
    if mocap_pred is None:
        hand_info = load_hand_info_from_pkl(mocap_path)
    else:
        hand_info = mocap_pred
    hand_bbox_list = hand_info['hand_bbox_list']
    mesh_bbox_list = extract_mesh_bbox_list(hand_info)
    IoU = calculate_box_iou(hand_bbox_list, mesh_bbox_list)
    filter_success = False
    if IoU >= IoU_thresh:
        filter_success = True
        if json_dict is not None:
            if vid_num in json_dict:
                json_dict[vid_num].append(frame_num)
            else:
                json_dict[vid_num] = [frame_num]

    if verbose:
        print(f'Filtering by IoU threshold successful: {filter_success}.')
    if json_path is not None:
        with open(json_path, 'w') as f:
            json.dump(json_dict, f)

    return IoU, filter_success


def create_rendered_image_dir_from_json(json_path, image_dir):
    os.makedirs(image_dir, exist_ok=True)
    with open(json_path, 'r') as f:
        json_dict = json.load(f)

    mocap_output_dir = join(DATA_DIR, 'valid', 'mocap_output')
    for vid_num in tqdm(
            json_dict,
            desc='Going through json to create rendered image directory'
    ):
        vid_dir = join(mocap_output_dir, vid_num)
        for frame_num in json_dict[vid_num]:
            frame_rendered_img_path = join(vid_dir, 'rendered', f'frame{frame_num}.jpg')
            dst_img_path = join(image_dir, f'vid{vid_num}_frame{frame_num}.jpg')
            shutil.copyfile(frame_rendered_img_path, dst_img_path)

if __name__ == '__main__':
    test_delta_hand_pose_generation = False
    test_mesh_bbox_extraction = False
    test_histogram = False
    test_filter_IoU = False
    test_contact = True

    if test_delta_hand_pose_generation:
        # Test delta hand pose vid generation
        generate_delta_hand_pose_vid(vid_num=0)

    if test_mesh_bbox_extraction:
        # Test mesh bbox extraction on mocap_output
        t0 = time.time()
        print('Testing mesh bbox extraction')
        vid_mocap_dir = join(
            '/home/junyao/Datasets/something_something_hand_demos',
            'pull_left', 'mocap_output', '2', 'mocap'
        )
        print(f'Processing mocap output from: {vid_mocap_dir}')
        from renderer.screen_free_visualizer import Visualizer
        visualizer = Visualizer('opendr')
        # visualizer = None
        for frame_num in tqdm(range(1, 39), desc='Going through frames..'):
            frame_pkl_path = join(vid_mocap_dir, f'frame{frame_num}_prediction_result.pkl')
            res_img_dir = join(
            '/home/junyao/Datasets/something_something_hand_demos',
            'pull_left', 'mocap_output', '2', 'hand_mesh_bbox'
            )
            os.makedirs(res_img_dir, exist_ok=True)
            res_img_path = join(res_img_dir, f'frame{frame_num}.jpg')
            visualize_frame_IoU(frame_pkl_path, res_img_path, visualizer)
        t1 = time.time()
        print(f'Done. Time elapsed: {t1 - t0:3f} seconds')

    if test_histogram:
        # Test histogram generation
        generate_frame_IoU_histogram(histogram_path='../mocap_output/ps_lr_train_valid_iou_hist.png')

    if test_filter_IoU:
        # Test filter data by IoU threshold
        IoU_thresh = 0.7
        filter_data_by_IoU_threshold(IoU_thresh=IoU_thresh,
                                     json_path=join(DATA_DIR, 'valid', f'IoU_{IoU_thresh}.json'))

        create_rendered_image_dir_from_json(json_path=join(DATA_DIR, 'valid', f'IoU_{IoU_thresh}.json'),
                                            image_dir=join(DATA_DIR, 'valid', f'IoU_{IoU_thresh}_rendered'))

    if test_contact:
        t0 = time.time()
        print('Testing mesh bbox extraction')

        from renderer.visualizer import Visualizer
        visualizer = Visualizer('opengl')
        # visualizer = None

        task = 'push_slightly'
        vid_nums = [1, 2, 3, 4, 5, 6]
        for vid_num in vid_nums:
            vid_mocap_dir = f'/home/junyao/Datasets/something_something_hand_demos/{task}/mocap_output/{vid_num}/mocap'
            vid_json_dir = f'/home/junyao/Datasets/something_something_hand_demos/{task}/bbs_json/{vid_num}'
            res_img_dir = f'/home/junyao/Datasets/something_something_hand_demos/{task}/mocap_output/{vid_num}/contact_rendered'
            print(f'Processing mocap output from: {vid_mocap_dir}')
            print(f'Processing bbs json from: {vid_json_dir}')

            frame_names = [fname.split('_')[0] for fname in os.listdir(vid_mocap_dir)]
            for frame_name in tqdm(frame_names, desc='Going through frames'):
                frame_pkl_path = join(vid_mocap_dir, f'{frame_name}_prediction_result.pkl')
                frame_json_path = join(vid_json_dir, f'{frame_name}.json')
                os.makedirs(res_img_dir, exist_ok=True)
                res_img_path = join(res_img_dir, f'{frame_name}.jpg')
                visualize_frame_contact(frame_pkl_path, frame_json_path, res_img_path, visualizer)
            t1 = time.time()
            print(f'Done. Time elapsed: {t1 - t0:3f} seconds')
