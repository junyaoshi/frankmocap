import json
import os
from os.path import join
from tqdm import tqdm

BBS_JSON_DIR = '/home/junyao/Datasets/something_something_processed/push_left_right/train/bbs_json'
FRAMES_DIR = '/home/junyao/Datasets/something_something_processed/push_left_right/train/frames'
TEST_JSON_DIR = join(BBS_JSON_DIR, str(0))

def process_json_from_cluster_for_cv(json_dir):
    json_files = [join(json_dir, f) for f in os.listdir(json_dir)]
    for json_file in tqdm(json_files, desc='Going through json files'):
        with open(json_file, 'r') as f:
            data = json.load(f)

        img_path = data['image_path']
        data['image_path'] = join(FRAMES_DIR, *img_path.split('/')[-2:])

        with open(json_file, 'w') as f:
            json.dump(data, f)

if __name__ == '__main__':
    # # debug test
    # process_json_from_cluster_for_cv(TEST_JSON_DIR)

    json_dirs = [join(BBS_JSON_DIR, d) for d in os.listdir(BBS_JSON_DIR)]
    for json_dir in tqdm(json_dirs, desc='Processing json directories...'):
        process_json_from_cluster_for_cv(json_dir)
