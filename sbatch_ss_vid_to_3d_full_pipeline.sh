#!/bin/bash
#SBATCH --mem-per-gpu=24G
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --cpus-per-gpu=4
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH -w node-3090-0
#SBATCH --job-name=ps_r
#SBATCH -o out/push_right_train_valid.out

export SS_JSON_DIR="/home/junyao/something_something_jsons"
export SS_VIDS_DIR="/Datasets/something_something/something_something"
export DATA_SAVE_DIR="/scratch/junyao/Datasets/something_something_processed"
export CONDA_ROOT="/scratch/junyao/anaconda3"
export TASK_NAME="push_right"
export IOU_THRESH=0.7

echo "SS_JSON_DIR: ${SS_JSON_DIR}"
echo "SS_VIDS_DIR: ${SS_VIDS_DIR}"
echo "DATA_SAVE_DIR: ${DATA_SAVE_DIR}"
echo "CONDA_ROOT: ${CONDA_ROOT}"
echo "TASK_NAME: ${TASK_NAME}"
echo "IOU_THRESH: ${IOU_THRESH}"

python /home/junyao/LfHV/frankmocap/ss_vid_to_3d_full_pipeline.py \
--ss_json_dir=${SS_JSON_DIR} \
--ss_vids_dir=${SS_VIDS_DIR} \
--data_save_dir=${DATA_SAVE_DIR} \
--conda_root=${CONDA_ROOT} \
--task_name=${TASK_NAME} \
--iou_thresh=${IOU_THRESH}

wait