#%%
import torch 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np 
import util 
import pickle 
import os 
os.chdir('/home/vss2134/gMVP')
WIDTH=64
class ProteinFeatureDataSet(Dataset):
    def __init__(self, feature_dir, meta_df_file):
        df = pd.read_csv(meta_df_file, sep = '\t')
        self.feature_dir = feature_dir
        self.meta_df = df
        self.n_samples =df.shape[0]
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx, :]
        transcript_id = row['transcript_id']
        transcript_feature_file = f'{self.feature_dir}/{transcript_id}.pickle'
        with open(transcript_feature_file, 'rb') as infile:
            transcript_features = pickle.load(infile)
        feature_len = transcript_features.shape[0]
        aa_pos = row['aa_pos']
        ref_aa = util.aa_index(row['ref_aa'])
        alt_aa = util.aa_index(row['alt_aa'])
        label = row['target']
        var_id = row['var']
        aa_pos -= 1 # guessing aa pos is 1 indexed 
        assert (ref_aa == transcript_features[aa_pos, 0]) 
        
        start = max(aa_pos - WIDTH, 0) 
        end = min(feature_len, aa_pos + 1 + WIDTH)
        var_start = start - (aa_pos - WIDTH)
        var_end = var_start + (end - start)
        var_feature = np.zeros([WIDTH * 2 + 1, transcript_features.shape[1]])
        var_feature[var_start:var_end] = transcript_features[start:end]
        mask = np.ones((WIDTH * 2 + 1, ), dtype=np.float32)
        mask[var_start:var_end] = 0.0
        
        return (torch.tensor(var_feature), 
                torch.tensor(ref_aa),
                torch.tensor(alt_aa),
                torch.tensor(label),
                var_id
        )
        
        
        
        
# %%
feature_dir = '/data/hz2529/zion/MVPContext/combined_feature_2021_v2'
md_file = 'example_metadata.tsv'


ds = ProteinFeatureDataSet(feature_dir, md_file)
ds[0]
# %%
