# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
import numpy as np
import cv2
import json
import torch
from torchvision.transforms import Normalize
from tqdm import tqdm

from demo.demo_options import DemoOptions
import mocap_utils.general_utils as gnu
import mocap_utils.demo_utils as demo_utils

from handmocap.hand_mocap_api import HandMocap
from handmocap.hand_bbox_detector import HandBboxDetector

import renderer.image_utils as imu
from renderer.viewer2D import ImShow
import time


def run_hand_mocap(args, bbox_detector, hand_mocap, visualizer):
    if args.save_pred_h5:
        raise NotImplementedError
    if args.input_dir is None:
        if args.hand_demos:
            pass
        input_paths = [args.input_path]
        assert args.out_dir is not None, "Please specify output dir to store the results"
        out_dirs = [args.out_dir]
        print(f'There are {len(input_paths)} total input paths.')
    else:
        assert args.out_parent_dir is not None, "Please specify output parent dir to store the results"
        input_paths = [osp.join(args.input_dir, d) for d in os.listdir(args.input_dir)]
        out_dirs = [osp.join(args.out_parent_dir, d) for d in os.listdir(args.input_dir)]
        print(f'There are {len(input_paths)} total input paths under \n{args.input_dir}.')

    n_processed, n_error, n_nohand, n_skipped = 0, 0, 0, 0
    for input_path, out_dir in tqdm(zip(input_paths, out_dirs), desc='Going through input paths...'):
        #Set up input data (images or webcam)
        args.input_path = input_path
        args.out_dir = out_dir
        try:
            input_type, input_data = demo_utils.setup_input(args)
        except AssertionError as e:
            n_error += len(os.listdir(input_path))
            print(f'Encountered AssertionError while processing: \n{input_path}. \nSkipping this input path.')
            print(e)
            continue

        assert args.out_dir is not None, "Please specify output dir to store the results"
        cur_frame = args.start_frame
        video_frame = 0

        while True:
            # load data
            load_bbox = False

            contact_list = None
            if input_type =='image_dir':
                if cur_frame < len(input_data):
                    image_path = input_data[cur_frame]
                    img_original_bgr  = cv2.imread(image_path)
                else:
                    img_original_bgr = None

            elif input_type == 'bbox_dir':
                if cur_frame < len(input_data):
                    # print("Use pre-computed bounding boxes")
                    image_path = input_data[cur_frame]['image_path']
                    hand_bbox_list = input_data[cur_frame]['hand_bbox_list']
                    body_bbox_list = input_data[cur_frame]['body_bbox_list']
                    contact_list = input_data[cur_frame]['contact_list']
                    img_original_bgr  = cv2.imread(image_path)
                    load_bbox = True
                else:
                    img_original_bgr = None

            elif input_type == 'video':
                _, img_original_bgr = input_data.read()
                if video_frame < cur_frame:
                    video_frame += 1
                    continue
                # save the obtained video frames
                image_path = osp.join(args.out_dir, "frames", f"{cur_frame:05d}.jpg")
                if img_original_bgr is not None:
                    video_frame += 1
                    if args.save_frame:
                        gnu.make_subdir(image_path)
                        cv2.imwrite(image_path, img_original_bgr)

            elif input_type == 'webcam':
                _, img_original_bgr = input_data.read()

                if video_frame < cur_frame:
                    video_frame += 1
                    continue
                # save the obtained video frames
                image_path = osp.join(args.out_dir, "frames", f"scene_{cur_frame:05d}.jpg")
                if img_original_bgr is not None:
                    video_frame += 1
                    if args.save_frame:
                        gnu.make_subdir(image_path)
                        cv2.imwrite(image_path, img_original_bgr)
            else:
                assert False, "Unknown input_type"

            cur_frame +=1
            if img_original_bgr is None or cur_frame > args.end_frame:
                break
            # print("--------------------------------------")

            # check for existing data and skip
            if args.save_pred_pkl:
                img_name = osp.basename(image_path)
                record = img_name.split('.')
                pkl_name = f"{'.'.join(record[:-1])}_prediction_result.pkl"
                pkl_path = osp.join(args.out_dir, 'mocap', pkl_name)
                if osp.exists(pkl_path):
                    n_skipped += 1
                    continue

            # bbox detection
            if load_bbox:
                body_pose_list = None
                raw_hand_bboxes = None
            elif args.crop_type == 'hand_crop':
                # hand already cropped, thererore, no need for detection
                img_h, img_w = img_original_bgr.shape[:2]
                body_pose_list = None
                raw_hand_bboxes = None
                hand_bbox_list = [ dict(right_hand = np.array([0, 0, img_w, img_h])) ]
            else:
                # Input images has other body part or hand not cropped.
                # Use hand detection model & body detector for hand detection
                assert args.crop_type == 'no_crop'
                detect_output = bbox_detector.detect_hand_bbox(img_original_bgr.copy())
                body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes = detect_output

            # save the obtained body & hand bbox to json file
            if args.save_bbox_output:
                demo_utils.save_info_to_json(args, image_path, body_bbox_list, hand_bbox_list)

            if len(hand_bbox_list) < 1:
                # print(f"No hand deteced: {image_path}")
                n_nohand += 1
                continue

            # Hand Pose Regression
            try:
                pred_output_list = hand_mocap.regress(
                        img_original_bgr, hand_bbox_list, add_margin=True)
            except:
                n_error += 1
                continue
            assert len(hand_bbox_list) == len(body_bbox_list)
            assert len(body_bbox_list) == len(pred_output_list)

            # extract mesh for rendering (vertices in image space and faces) from pred_output_list
            pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

            # visualize
            res_img = visualizer.visualize(
                img_original_bgr,
                pred_mesh_list = pred_mesh_list,
                hand_bbox_list = hand_bbox_list
            )

            # show result in the screen
            if not args.no_display:
                res_img = res_img.astype(np.uint8)
                ImShow(res_img)

            # save the image (we can make an option here)
            if args.out_dir is not None:
                demo_utils.save_res_img(args.out_dir, image_path, res_img)

            # save predictions to pkl
            if args.save_pred_pkl:
                demo_type = 'hand'
                demo_utils.save_pred_to_pkl(
                    args, demo_type, image_path,
                    body_bbox_list, hand_bbox_list, contact_list, pred_output_list
                )

            n_processed += 1
            # print(f"Processed : {image_path}")

        #save images as a video
        if not args.no_video_out and input_type in ['video', 'webcam']:
            demo_utils.gen_video_out(args.out_dir, args.seq_name)

        # When everything done, release the capture
        if input_type =='webcam' and input_data is not None:
            input_data.release()
        cv2.destroyAllWindows()

    print(f'Converted bounding boxes to 3D hand poses. '
          f'Total frames: {n_processed + n_error + n_nohand + n_skipped}; '
          f'Processed frames: {n_processed}; '
          f'Error frames: {n_error}; '
          f'No hand frames: {n_nohand}; '
          f'Skipped frames: {n_skipped}.')

  
def main():
    args = DemoOptions().parse()
    args.use_smplx = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    #Set Bbox detector
    bbox_detector =  HandBboxDetector(args.view_type, device)

    # Set Mocap regressor
    hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device = device)

    # Set Visualizer
    print(f'Setting up {args.renderer_type} visualizer...')
    if args.renderer_type == 'None':
        visualizer = None  # debug mode
    else:
        if args.renderer_type in ['pytorch3d', 'opendr']:
            from renderer.screen_free_visualizer import Visualizer
        else:
            from renderer.visualizer import Visualizer
        visualizer = Visualizer(args.renderer_type)

    # run
    run_hand_mocap(args, bbox_detector, hand_mocap, visualizer)
   

if __name__ == '__main__':
    main()
