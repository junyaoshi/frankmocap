import json
import os
from os.path import join
import pickle

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as im
import cv2
from torchvision.ops import box_iou
import torch

import mocap_utils.demo_utils as demo_utils
from renderer.image_utils import draw_hand_bbox
from renderer.screen_free_visualizer import Visualizer

DATA_DIR = '/home/junyao/Datasets/something_something_new_3d'


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
    return pickle.load(open(pkl_path, 'rb'))


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


def visualize_frame_IoU(frame_pkl_path, res_img_path):
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


if __name__ == '__main__':
    # # Test delta hand pose vid generation
    # generate_delta_hand_pose_vid(vid_num=0)

    # # Test mesh bbox extraction on ss
    # vid_num = 200
    # ss_vid_dir = join(DATA_DIR, str(vid_num))
    # vid_mocap_dir = join(ss_vid_dir, 'mocap')
    # frame_num = 0
    # frame_pkl_path = join(vid_mocap_dir, f'frame{frame_num}_prediction_result.pkl')
    # hand_info = load_hand_info_from_pkl(frame_pkl_path)
    # extract_mesh_bbox(hand_info)

    # Test mesh bbox extraction on mocap_output
    vid_num = 0
    vid_mocap_dir = join('..', 'mocap_output_3rd', str(vid_num), 'mocap')
    visualizer = Visualizer('opendr')
    for frame_num in tqdm(range(1, 39), desc='Going through frames..'):
        frame_pkl_path = join(vid_mocap_dir, f'frame{frame_num}_prediction_result.pkl')
        res_img_dir = join('..', 'mocap_output_3rd', str(vid_num), 'hand_mesh_bbox')
        os.makedirs(res_img_dir, exist_ok=True)
        res_img_path = join(res_img_dir, f'frame{frame_num}.jpg')
        visualize_frame_IoU(frame_pkl_path, res_img_path)
