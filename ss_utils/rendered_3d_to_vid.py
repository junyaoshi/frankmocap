import cv2
import os
from os.path import join
from tqdm import tqdm

task = 'push_slightly'
data_dir = f'/home/junyao/Datasets/something_something_hand_demos/{task}'
video_nums = [1, 2, 3, 4, 5, 6]
print(f'There are {len(video_nums)} video directories')

for video_num in tqdm(video_nums, desc='Going through video directories...'):
    rendered_imgs_dir = join(data_dir, 'mocap_output', str(video_num), 'contact_rendered')
    rendered_vid_dir = join(data_dir, 'contact_rendered_vid')
    os.makedirs(rendered_vid_dir, exist_ok=True)
    rendered_vid_name = f'rendered_{video_num}.mp4'

    images = [img for img in os.listdir(rendered_imgs_dir) if img.endswith(".jpg")]
    images = sorted(images, key=lambda x: int(x[5:].split('.')[0]))
    frame = cv2.imread(join(rendered_imgs_dir, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 3
    video = cv2.VideoWriter(join(rendered_vid_dir, rendered_vid_name), fourcc, fps, (width,height))

    for image in images:
        video.write(cv2.imread(join(rendered_imgs_dir, image)))

    cv2.destroyAllWindows()
    video.release()
