import json
import os
from os.path import join
from tqdm import tqdm

json_dir = '/home/junyao/Datasets/something_something_new_jsons/0'

def process_json_from_cluster_for_cv(json_dir):
    json_files = [join(json_dir, f) for f in os.listdir(json_dir)]
    for json_file in tqdm(json_files, desc='Going through json files'):
        with open(json_file, 'r') as f:
            data = json.load(f)

        img_path = data['image_path']
        data['image_path'] = '/home/junyao/Datasets' + img_path[22:]

        with open(json_file, 'w') as f:
            json.dump(data, f)

if __name__ == '__main__':
    process_json_from_cluster_for_cv(json_dir)
