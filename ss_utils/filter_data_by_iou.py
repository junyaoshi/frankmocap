from filter_utils import filter_data_by_IoU_threshold
from os.path import join

DATA_HOME_DIR = '/home/junyao/Datasets/something_something_processed'
SPLITS = ['train']
IOU_THRESH = 0.7
TASK_NAMES = ['push_left_right']

for task_name in TASK_NAMES:
    task_data_dir = join(DATA_HOME_DIR, task_name)
    for split in SPLITS:
        split_data_dir = join(task_data_dir, split)
        filter_data_by_IoU_threshold(
            data_dir=split_data_dir,
            IoU_thresh=IOU_THRESH,
            json_path=join(split_data_dir, f'IoU_{IOU_THRESH}.json')
        )
