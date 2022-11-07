import os
import os.path as osp
import time
from collections import OrderedDict
import sys
import json

import torch
import cv2
import numpy as np

from handmocap.hand_mocap_api import HandMocap
from mocap_utils import demo_utils
import mocap_utils.general_utils as gnu


def __get_input_type(input_path):
    image_exts = ('jpg', 'png', 'jpeg', 'bmp')
    video_exts = ('mp4', 'avi', 'mov')
    extension = osp.splitext(input_path)[1][1:]

    if extension.lower() in video_exts:
        input_type = 'video'
    elif extension.lower() == 'json':
        input_type = 'json_file'
    elif extension.lower() in image_exts:
        input_type = 'image_file'
    elif osp.isdir(input_path):
        raise TypeError(f'Input path {input_path} is a directory.')
    elif input_path == 'webcam':
        input_type = 'webcam'
    else:
        assert False, f"Unknown input path {input_path}. It should be an image," + \
                      "or an image folder, or a video file, or \'webcam\' "
    return extension, input_type


def __img_seq_setup(out_dir, save_pred_pkl, save_pred_vis):
    if save_pred_pkl:
        mocap_out_dir = osp.join(out_dir, "mocap")
        os.makedirs(mocap_out_dir, exist_ok=True)
    if save_pred_vis:
        render_out_dir = osp.join(out_dir, 'rendered')
        os.makedirs(render_out_dir, exist_ok=True)


def setup_path_input(input_path, out_dir, save_pred_pkl, save_pred_vis):
    # get type of input
    extension, input_type = __get_input_type(input_path)

    if input_type == 'video' or input_type == 'webcam':
        raise TypeError(f'Unsupported input type: {input_type}')
    elif input_type == 'image_file':
        raise NotImplementedError
    elif input_type == 'json_file':
        __img_seq_setup(out_dir, save_pred_pkl, save_pred_vis)
        image_path, body_bbox_list, hand_bbox_list, contact_list = demo_utils.load_info_from_json(input_path)
        input_data = dict(
            image_path=image_path,
            hand_bbox_list=hand_bbox_list,
            body_bbox_list=body_bbox_list,
            contact_list=contact_list
        )
        return input_type, input_data
    else:
        raise TypeError(f"Unknown input type: {input_type}")


def load_info_from_dict(input_dict):
    data = input_dict
    # image path
    assert ('image_path' in data), "Path of input image should be specified"
    image_path = data['image_path']
    assert osp.exists(image_path), f"{image_path} does not exists"

    # body bboxes
    body_bbox_list = list()
    if 'body_bbox_list' in data:
        body_bbox_list = data['body_bbox_list']
        assert isinstance(body_bbox_list, list)
        for b_id, body_bbox in enumerate(body_bbox_list):
            if isinstance(body_bbox, list) and len(body_bbox) == 4:
                body_bbox_list[b_id] = np.array(body_bbox)

    # hand bboxes
    hand_bbox_list = list()
    if 'hand_bbox_list' in data:
        hand_bbox_list = data['hand_bbox_list']
        assert isinstance(hand_bbox_list, list)
        for hand_bbox in hand_bbox_list:
            for hand_type in ['left_hand', 'right_hand']:
                if hand_type in hand_bbox:
                    bbox = hand_bbox[hand_type]
                    if isinstance(bbox, list) and len(bbox) == 4:
                        hand_bbox[hand_type] = np.array(bbox)
                    else:
                        hand_bbox[hand_type] = None

    # contact
    contact_list = []
    if 'contact_list' in data:
        contact_list = data['contact_list']
        assert isinstance(contact_list, list)

    return image_path, body_bbox_list, hand_bbox_list, contact_list


def setup_dict_input(input_dict, out_dir, save_pred_pkl, save_pred_vis):
    __img_seq_setup(out_dir, save_pred_pkl, save_pred_vis)
    image_path, body_bbox_list, hand_bbox_list, contact_list = load_info_from_dict(input_dict)
    input_data = dict(
        image_path=image_path,
        hand_bbox_list=hand_bbox_list,
        body_bbox_list=body_bbox_list,
        contact_list=contact_list
    )
    return input_data


def create_pred_dict(
        demo_type,
        image_path,
        image_shape,
        body_bbox_list,
        hand_bbox_list,
        contact_list,
        pred_output_list,
        use_smplx=True,
        save_mesh=True
):
    smpl_type = 'smplx' if use_smplx else 'smpl'
    assert demo_type in ['hand', 'body', 'frank']
    if demo_type in ['hand', 'frank']:
        assert smpl_type == 'smplx'

    assert len(hand_bbox_list) == len(body_bbox_list)
    assert len(body_bbox_list) == len(pred_output_list)

    # demo type / smpl type / image / bbox
    saved_data = OrderedDict()
    saved_data['demo_type'] = demo_type
    saved_data['smpl_type'] = smpl_type
    saved_data['image_path'] = osp.abspath(image_path)
    saved_data['body_bbox_list'] = body_bbox_list
    saved_data['hand_bbox_list'] = hand_bbox_list
    if contact_list is not None:
        saved_data['contact_list'] = contact_list
    saved_data['save_mesh'] = save_mesh
    saved_data['image_shape'] = image_shape

    saved_data['pred_output_list'] = list()
    num_subject = len(hand_bbox_list)
    for s_id in range(num_subject):
        # predict params
        pred_output = pred_output_list[s_id]
        if pred_output is None:
            saved_pred_output = None
        else:
            saved_pred_output = dict()
            if demo_type == 'hand':
                for hand_type in ['left_hand', 'right_hand']:
                    pred_hand = pred_output[hand_type]
                    saved_pred_output[hand_type] = dict()
                    saved_data_hand = saved_pred_output[hand_type]
                    if pred_hand is not None:
                        for pred_key in pred_hand:
                            if pred_key.find("vertices") < 0 or pred_key == 'faces':
                                saved_data_hand[pred_key] = pred_hand[pred_key]
                            elif save_mesh:
                                if pred_key != 'faces':
                                    saved_data_hand[pred_key] = pred_hand[pred_key].astype(np.float16)
                                else:
                                    saved_data_hand[pred_key] = pred_hand[pred_key]
            else:
                for pred_key in pred_output:
                    if pred_key.find("vertices") < 0 or pred_key == 'faces':
                        saved_pred_output[pred_key] = pred_output[pred_key]
                    else:
                        if save_mesh:
                            if pred_key != 'faces':
                                saved_pred_output[pred_key] = \
                                    pred_output[pred_key].astype(np.float16)
                            else:
                                saved_pred_output[pred_key] = pred_output[pred_key]

        saved_data['pred_output_list'].append(saved_pred_output)

    return saved_data


def save_pred_to_pkl(pred_dict, out_dir, image_path, verbose=False):
    """write data to pkl"""
    img_name = osp.basename(image_path)
    record = img_name.split('.')
    pkl_name = f"{'.'.join(record[:-1])}_prediction_result.pkl"
    pkl_path = osp.join(out_dir, 'mocap', pkl_name)
    gnu.make_subdir(pkl_path)
    gnu.save_pkl(pkl_path, pred_dict)
    if verbose:
        print(f"Prediction saved: {pkl_path}")


def setup_handmocap(
        frankmocap_dir='/home/junyao/LfHV/frankmocap',
        checkpoint_hand_relative="extra_data/hand_module/pretrained_weights/pose_shape_best.pth",
        smpl_dir_relative="extra_data/smpl/",
):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    # Set up mocap regressor
    print('Loading frankmocap hand mocap...')
    cwd = os.getcwd()
    checkpoint_hand = osp.join(frankmocap_dir, checkpoint_hand_relative)
    smpl_dir = osp.join(frankmocap_dir, smpl_dir_relative)
    sys.path.insert(1, frankmocap_dir)
    os.chdir(frankmocap_dir)
    hand_mocap = HandMocap(checkpoint_hand, smpl_dir, device=device)
    os.chdir(cwd)

    return hand_mocap


def setup_visualizer(renderer_type='opengl'):
    # Set Visualizer
    print(f'Setting up {renderer_type} visualizer...')
    if renderer_type == 'None':
        visualizer = None  # debug mode
    else:
        if renderer_type in ['pytorch3d', 'opendr']:
            from renderer.screen_free_visualizer import Visualizer
        else:
            from renderer.visualizer import Visualizer
        visualizer = Visualizer(renderer_type)

    return visualizer


def run_frame_hand_mocap(
        input_path,
        out_dir,
        hand_mocap,
        visualizer,
        bbox_dict=None,
        img_original_bgr=None,
        save_pred_pkl=True,
        save_pred_vis=True,
        use_smplx=True,
        save_mesh=True,
        verbose=True
):
    assert input_path is not None or bbox_dict is not None, 'Need to provide either input path or bbox dict.'
    if save_pred_vis:
        assert visualizer is not None

    t0 = time.time()
    pred_dict, success = None, False

    if bbox_dict is None:
        try:
            input_type, input_data = setup_path_input(input_path, out_dir, save_pred_pkl, save_pred_vis)
        except Exception as e:
            print(f'FrankMocap encountered error while setting up: {input_path}.')
            print(e)
            return pred_dict, success
        assert input_type == 'json_file', "Currently only supported for bbox json file input"
    else:
        try:
            input_data = setup_dict_input(bbox_dict, out_dir, save_pred_pkl, save_pred_vis)
        except Exception as e:
            print(f'FrankMocap encountered error while setting up given bbox dict.')
            print(e)
            return pred_dict, success

    image_path = input_data['image_path']
    hand_bbox_list = input_data['hand_bbox_list']
    body_bbox_list = input_data['body_bbox_list']
    contact_list = None
    if 'contact_list' in input_data:
        contact_list = input_data['contact_list']
    if img_original_bgr is None:
        img_original_bgr = cv2.imread(image_path)
    image_shape = np.array(img_original_bgr.shape)[:2]

    assert len(hand_bbox_list) >= 1, f"No hand deteced: \n{image_path}"

    # Hand Pose Regression
    try:
        pred_output_list = hand_mocap.regress(
            img_original_bgr, hand_bbox_list, add_margin=True
        )
    except Exception as e:
        print(f'FrankMocap encountered error while running hand mocap regression for: '
              f'{input_path if bbox_dict is None else "given bbox dict"}')
        print(e)
        return pred_dict, success

    assert len(hand_bbox_list) == len(body_bbox_list)
    assert len(body_bbox_list) == len(pred_output_list)

    # extract mesh for rendering (vertices in image space and faces) from pred_output_list
    pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

    # visualize
    if save_pred_vis:
        res_img = visualizer.visualize(
            img_original_bgr,
            pred_mesh_list=pred_mesh_list,
            hand_bbox_list=hand_bbox_list
        )
        demo_utils.save_res_img(out_dir, image_path, res_img, verbose)

    pred_dict = create_pred_dict(
        demo_type='hand',
        image_path=image_path,
        image_shape=image_shape,
        body_bbox_list=body_bbox_list,
        hand_bbox_list=hand_bbox_list,
        contact_list=contact_list,
        pred_output_list=pred_output_list,
        use_smplx=use_smplx,
        save_mesh=save_mesh
    )

    # save predictions to pkl
    if save_pred_pkl:
        save_pred_to_pkl(pred_dict, out_dir, image_path, verbose)

    t1 = time.time()
    success = True
    if verbose:
        print(f'Converted bounding box at {input_path if bbox_dict is None else "given bbox dict"}.')
        print(f'FrankMocap time elapsed: {t1 - t0:.3f} seconds.')

    return pred_dict, success


if __name__ == '__main__':
    input_path = '/home/junyao/test/frame10.json'
    out_dir = '/home/junyao/test'
    renderer_type = 'None'
    save_pred_vis = False
    save_pred_pkl = True
    load_from_json = True

    hand_mocap = setup_handmocap()
    visualizer = setup_visualizer(renderer_type=renderer_type)

    if load_from_json:
        with open(input_path, 'r') as f:
            bbox_dict = json.load(f)

    pred_dict, success = run_frame_hand_mocap(
        input_path=None,
        out_dir=out_dir,
        bbox_dict=bbox_dict,
        hand_mocap=hand_mocap,
        visualizer=visualizer,
        save_pred_pkl=save_pred_pkl,
        save_pred_vis=save_pred_vis
    )

    print('Done.')
