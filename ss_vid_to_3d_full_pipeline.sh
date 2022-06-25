export SS_JSON_DIR="/home/junyao/Datasets/something_something_original"
export SS_VIDS_DIR="/home/junyao/Datasets/something_something_original/something_something"
export DATA_SAVE_DIR="/home/junyao/Datasets/something_something_processed"
export CONDA_ROOT="/home/junyao/anaconda3"
export TASK_NAME="move_down"
export IOU_THRESH=0.7

echo "SS_JSON_DIR: ${SS_JSON_DIR}"
echo "SS_VIDS_DIR: ${SS_VIDS_DIR}"
echo "DATA_SAVE_DIR: ${DATA_SAVE_DIR}"
echo "CONDA_ROOT: ${CONDA_ROOT}"
echo "TASK_NAME: ${TASK_NAME}"
echo "IOU_THRESH: ${IOU_THRESH}"

CUDA_VISIBLE_DEVICES=1 python /home/junyao/LfHV/frankmocap/ss_vid_to_3d_full_pipeline.py \
--ss_json_dir=${SS_JSON_DIR} \
--ss_vids_dir=${SS_VIDS_DIR} \
--data_save_dir=${DATA_SAVE_DIR} \
--conda_root=${CONDA_ROOT} \
--task_name=${TASK_NAME} \
--iou_thresh=${IOU_THRESH} \
--run_on_cv_server