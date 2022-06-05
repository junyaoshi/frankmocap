import json, pickle, os
from os.path import join
from tqdm import tqdm

bb_pkl_path = '/home/junyao/LfHV/hand_object_detector/bb_pkl/200.pkl'
bbs = pickle.load(open(bb_pkl_path, 'rb'))

for frame in tqdm(bbs, desc='Going through bounding boxes...'):
    if bbs[frame] is None:
        print(f'No bounding box detected for {frame}.')
        continue
    js = {"image_path": join('/home/junyao/Datasets/something_something', str(200), f'{frame}.jpg'),
          "body_bbox_list": [[]]}
    if len(bbs[frame]) == 1:
        hand = [int(bbs[frame][0][0]), int(bbs[frame][0][1]),
                int(bbs[frame][0][2] - bbs[frame][0][0]),
                int(bbs[frame][0][3] - bbs[frame][0][1])]
        if bbs[frame][0][9] == 0:
            js["hand_bbox_list"] = [{"left_hand": hand,
                                     "right_hand": [],
                                     }]
        elif bbs[frame][0][9] == 1:
            js["hand_bbox_list"] = [{"left_hand": [],
                                     "right_hand": hand,
                                     }]
        else:
            print("error")

    elif len(bbs[frame]) == 2:
        hand1 = [int(bbs[frame][0][0]), int(bbs[frame][0][1]),
                 int(bbs[frame][0][2] - bbs[frame][0][0]),
                 int(bbs[frame][0][3] - bbs[frame][0][1])]
        hand2 = [int(bbs[frame][1][0]), int(bbs[frame][1][1]),
                 int(bbs[frame][1][2] - bbs[frame][1][0]),
                 int(bbs[frame][1][3] - bbs[frame][1][1])]
        if bbs[frame][0][9] == 0:
            js["hand_bbox_list"] = [{"left_hand": hand1,
                                     "right_hand": hand2,
                                     }]
        elif bbs[frame][0][9] == 1:
            js["hand_bbox_list"] = [{"left_hand": hand2,
                                     "right_hand": hand1,
                                     }]
        else:
            print("error")
    else:
        print("error")

    json_dir = join('/home/junyao/LfHV/hand_object_detector', 'bb_json', str(200))
    os.makedirs(json_dir, exist_ok=True)
    with open(join(json_dir, f'{frame}.json'), 'w') as outfile:
        json.dump(js, outfile)
