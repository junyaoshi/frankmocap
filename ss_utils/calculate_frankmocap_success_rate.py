import json
import os
from os.path import join
from tqdm import tqdm

ss_dir = '/home/junyao/Datasets/something_something'  # something-something
fm_dir = '/home/junyao/Datasets/something_something_new_3d'  # frankmocap


if __name__ == '__main__':
    ss_subdirs = os.listdir(ss_dir)
    print(f'There are {len(ss_subdirs)} something-something pushing videos')
    ss_n_images = 0
    for ss_subdir in tqdm(ss_subdirs, desc='Going through s-s subdirectories...'):
        images = [img for img in os.listdir(join(ss_dir, ss_subdir)) if img.endswith(".jpg")]
        ss_n_images += len(images)
    print(f'There are {ss_n_images} something-something pushing images')

    fm_subdirs = os.listdir(fm_dir)
    print(f'There are {len(fm_subdirs)} frank mocap pushing videos')
    fm_n_images = 0
    for fm_subdir in tqdm(fm_subdirs, desc='Going through frank mocap subdirectories...'):
        images = [img for img in os.listdir(join(fm_dir, fm_subdir, 'rendered')) if img.endswith(".jpg")]
        fm_n_images += len(images)
    print(f'There are {fm_n_images} frank mocap pushing images')

    print(f'Frank mocap video success rate is {len(fm_subdirs)/len(ss_subdirs):.4f}')
    print(f'Frank mocap image success rate is {fm_n_images/ ss_n_images:.4f}')
