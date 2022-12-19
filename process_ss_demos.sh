export DEMOS_DIR="/home/junyao/Datasets/something_something_hand_demos"
export DEMO_TYPE="hand_demos"
export CONDA_ROOT="/home/junyao/anaconda3"
export IOU_THRESH=0.6

echo "DEMOS_DIR: ${DEMOS_DIR}"
echo "DEMO_TYPE: ${DEMO_TYPE}"
echo "CONDA_ROOT: ${CONDA_ROOT}"
echo "IOU_THRESH: ${IOU_THRESH}"

CUDA_VISIBLE_DEVICES=0 python /home/junyao/LfHV/frankmocap/process_ss_demos.py \
--demos_dir=${DEMOS_DIR} \
--demo_type=${DEMO_TYPE} \
--conda_root=${CONDA_ROOT} \
--iou_thresh=${IOU_THRESH} \
--save_contact \
--save_savgol \
