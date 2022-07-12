export HAND_DEMO_DIR="/home/junyao/Datasets/something_something_hand_demos"
export CONDA_ROOT="/home/junyao/anaconda3"
export IOU_THRESH=0.7

echo "HAND_DEMO_DIR: ${HAND_DEMO_DIR}"
echo "CONDA_ROOT: ${CONDA_ROOT}"
echo "IOU_THRESH: ${IOU_THRESH}"

CUDA_VISIBLE_DEVICES=1 python /home/junyao/LfHV/frankmocap/process_ss_hand_demos.py \
--hand_demos_dir=${HAND_DEMO_DIR} \
--conda_root=${CONDA_ROOT} \
--iou_thresh=${IOU_THRESH} \
