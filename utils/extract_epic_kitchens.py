
import lmdb
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

print(
"""
Please first clone 'https://github.com/epic-kitchens/C2-Action-Detection' and 
download the features with 'C2-Action-Detection/BMNProposalGenerator/scripts/download_data_ek100_full.sh'.
After it, please enter the paths to the repo and features below.
"""
)

repo_path = '...'
rgb_lmdb_path = '...'
flow_lmdb_path = '...'
output_path = 'data/epic-kitchens/features'

dataset_path = os.path.join(repo_path, 'BMNProposalGenerator/data/ek100')

for sset in ['training', 'validation']:

    annotations = pd.read_csv(os.path.join(dataset_path, sset + '.csv'), names=['id', 'video', 'start', 'stop', 'verb', 'noun', 'action'], index_col='id')
    if isinstance(annotations.iloc[0]['start'], str):
        annotations = pd.read_csv(os.path.join(dataset_path, sset + '.csv'), index_col='narration_id')

    video_list = [v.strip() for v in annotations['video'].unique()]
    lengths = pd.read_csv(os.path.join(dataset_path, 'video_lengths.csv'))
    length_dict = lengths = lengths.set_index('video').to_dict()['frames']

    fname_template='frame_{:010d}.jpg'
    env_rgb = lmdb.open(rgb_lmdb_path, readonly=True, lock=False)
    env_flow = lmdb.open(flow_lmdb_path, readonly=True, lock=False)

    def _read(e, fname):
        # with env_rgb.begin() as e:
        dd = e.get(fname.encode())
        if dd is None:
            raise ValueError(fname)
        dd = np.frombuffer(dd, dtype='float32').reshape(-1, 1)
        return dd

    with env_rgb.begin() as rgb_data:
        with env_flow.begin() as flow_data:
            for video_name in tqdm(video_list):
                length = length_dict[video_name]

                feats = []
                ff = np.arange(length-1)+1
                for f in ff:
                    fname = video_name + '_' + fname_template.format(f)
                    rgb = _read(rgb_data, fname)
                    flow = _read(flow_data, fname)
                    feat = np.concatenate([rgb, flow])
                    feats.append(feat)
                video_data = np.hstack(feats).T # T, H
        
                savefname = os.path.join(output_path, video_name + ".npy")
                np.save(savefname, video_data)
