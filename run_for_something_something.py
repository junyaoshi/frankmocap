import subprocess
from tqdm import tqdm

vid_nums = [200]

for vid_num in tqdm(vid_nums, desc='Going through videos...'):
    print(f'Running Frank Mocap on video {vid_num}')
    bashCommand = f'xvfb-run -a python -m demo.demo_handmocap ' \
                  f'--input_path /home/junyao/Datasets/something_something_new_jsons/{vid_num} ' \
                  f'--out_dir /home/junyao/LfHV/frankmocap/mocap_output/200 ' \
                  f'--save_pred_pkl --save_mesh'
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    out, error = process.communicate()
    print(out, error)
