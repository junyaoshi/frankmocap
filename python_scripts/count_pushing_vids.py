import json
from os.path import join
from tqdm import tqdm

_DATA_DIR = '/home/junyao/Datasets/something_something_original'
_TEMPLATES = {
    # "Closing [something]": 0,
    # "Moving [something] away from the camera": 1,
    # "Moving [something] towards the camera": 2,
    # "Opening [something]": 3,
    # "Pushing [something] from left to right": 4,
    # "Pushing [something] from right to left": 5,
    # "Poking [something] so lightly that it doesn't or almost doesn't move": 6,
    "Moving [something] down": 7,
    "Moving [something] up": 8,
    # "Pulling [something] from left to right": 9,
    # "Pulling [something] from right to left": 10,
    # "Pushing [something] with [something]": 11,
    # "Moving [something] closer to [something]": 12,
    # "Plugging [something] into [something]": 13,
    # "Pushing [something] so that it slightly moves": 14
}

if __name__ == '__main__':
    # load json
    train_list = json.load(open(join(_DATA_DIR, 'something-something-v2-train.json'), 'r'))
    valid_list = json.load(open(join(_DATA_DIR, 'something-something-v2-validation.json'), 'r'))

    # split generator
    train_dict, valid_dict = {}, {}
    for k in _TEMPLATES.keys():
        train_dict[k] = []
        valid_dict[k] = []
    for train_data in tqdm(train_list, desc='Parsing training json'):
        if train_data['template'] in _TEMPLATES.keys():
            train_dict[train_data['template']].append(train_data['id'])
    for valid_data in tqdm(valid_list, desc='Parsing validation json'):
        if valid_data['template'] in _TEMPLATES.keys():
            valid_dict[valid_data['template']].append(valid_data['id'])

    # print and visualize
    for k, v in train_dict.items():
        print(f'Train | {k} : {len(v)}')
    for k, v in valid_dict.items():
        print(f'Valid | {k} : {len(v)}')
    train_n_vids = sum([len(v) for v in train_dict.values()])
    print(f'There are a total of {train_n_vids} pushing videos in training set')
    valid_n_vids = sum([len(v) for v in valid_dict.values()])
    print(f'There are a total of {valid_n_vids} pushing videos in validation set')
    print(f'There are a total of {train_n_vids + valid_n_vids} pushing videos')
