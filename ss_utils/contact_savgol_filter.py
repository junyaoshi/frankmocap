import os
import os.path as osp
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm

from ss_utils.filter_utils import determine_which_hand


if __name__ == '__main__':
    mocap_dirs = [
        '/home/junyao/Datasets/something_something_processed/push_left/valid/mocap_output/85114/mocap',
        '/home/junyao/Datasets/something_something_processed/push_left/valid/mocap_output/31379/mocap',
        # '/home/junyao/Datasets/something_something_processed/push_left/valid/mocap_output/189745/mocap',
        # '/home/junyao/Datasets/something_something_processed/push_left/valid/mocap_output/29583/mocap',
        # '/home/junyao/Datasets/something_something_processed/push_left/valid/mocap_output/206696/mocap',
        # '/home/junyao/Datasets/something_something_processed/push_left/valid/mocap_output/188257/mocap',
        # '/home/junyao/Datasets/something_something_processed/push_left/valid/mocap_output/173389/mocap',
        # '/home/junyao/Datasets/something_something_processed/push_left/valid/mocap_output/161067/mocap',
        # '/home/junyao/Datasets/something_something_processed/move_up/valid/mocap_output/12499/mocap',
        # '/home/junyao/Datasets/something_something_processed/move_up/valid/mocap_output/18846/mocap',
        # '/home/junyao/Datasets/something_something_processed/move_up/valid/mocap_output/12033/mocap',
    ]
    contact_plot_dir = 'contact_plots'
    os.makedirs(contact_plot_dir, exist_ok=True)
    windows = [
        # 3,
        5,
    ]
    degrees = [
        # 0,
        1,
        # 2,
    ]

    for i, mocap_dir in enumerate(mocap_dirs):
        frame_nums = sorted([int(p.split('_')[0][5:]) for p in os.listdir(mocap_dir)])
        contacts = []
        print(f'\nProcessing {len(frame_nums)} frames at: {mocap_dir}')
        for frame_num in tqdm(frame_nums):
            mocap_path = osp.join(mocap_dir, f'frame{frame_num}_prediction_result.pkl')
            with open(mocap_path, 'rb') as f:
                hand_info = pickle.load(f)

            hand = determine_which_hand(hand_info)
            contact_state = hand_info['contact_list'][0][hand]
            contact = 1 if contact_state == 3 else 0
            contacts.append(contact)

        print(f'{"Original trajectory: ":<30}{str(np.array(contacts)):>50}')
        for window in windows:
            for degree in degrees:
                if degree >= window - 1:
                    continue
                contacts_filtered = savgol_filter(contacts, window_length=window, polyorder=degree)
                contacts_filtered = (contacts_filtered >= 0.5).astype(np.int64)
                filter_str = f'Filtered window {window} degree {degree}: '
                print(f'{filter_str:<30}{str(contacts_filtered):>50}')

                fig, (ax1, ax2) = plt.subplots(2, 1)
                ax1.step(range(len(contacts)), contacts)
                ax1.set_ylim(-1, 2)
                ax2.step(range(len(contacts_filtered)), contacts_filtered)
                ax2.set_ylim(-1, 2)
                plt.savefig(osp.join(contact_plot_dir, f'savgol_img{i}_window{window}_deg{degree}.png'))
                plt.show()

    savgol_params_path = 'savgol_params.pkl'
    savgol_params = {'window_length': 5, 'polyorder': 1}
    with open(savgol_params_path, 'wb') as f:
        pickle.dump(savgol_params, f)
