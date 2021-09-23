#%%
import random
import pandas as pd
import argparse
import util
import traceback
import numpy as np
import json
import os
import pickle

from multiprocessing import Process

#version in 2021

random.seed(2020)

base_dir = '/data/hz2529/zion/MVPContext/'
feature_dir = f'{base_dir}/feature'
output_dir = '/home/vss2134/gMVP/dataset/build_feature_out'

##ADDED
#os.chdir('/data/hz2529/zion/MVPContext/')
##

def read_compara(path):
    res = []
    perc_id = []
    with open(path) as f:
        for line in f:
            name, _perc_id, seq = line.strip().split('\t')
            if name == 'homosapien':
                _perc_id = 100.0
            else:  
                _perc_id = float(_perc_id)
            seq = [util.aa_index(a) for a in seq]
            res.append(seq)
            perc_id.append(_perc_id)
    perc_id, msa = np.tile(np.array(perc_id),
                           [len(res[0]), 1]), np.array(res).transpose()
    return np.concatenate([msa, perc_id], axis=-1)


def calc_angle(x):
    x = x / 180.0 * np.pi
    return np.stack([np.sin(x), np.cos(x)], axis=-1)


def read_netsurfp2(netsurfp2_path):
    res = []
    with open(netsurfp2_path, 'r') as fr:
        data = json.load(fr)
    names = ['q3_prob', 'rsa', 'interface', 'disorder', 'phi', 'psi']
    for n in names:
        fea = np.array(data[n])
        if n in ['phi', 'psi']:
            fea = calc_angle(fea)
        if fea.ndim == 1:
            fea = np.expand_dims(fea, axis=1)
        res.append(fea)
    res = np.concatenate(res, axis=-1)
    return res


def read_region(region_path):
    res = np.load(region_path)
    return np.nan_to_num(res)


'''
feature schema
[0,1) sequence
[1,21) conservation from hhblits
[21,41) conservation from compara 
#[41,61) conservation from hhblits, id=90, diff=0
#[61,81) conservation from hhblits, id=99, diff=0
[81,481) sequences of 400 species
[481,491) predicted structural features using NetSurfp2
[491,495) observed and estimated number of missense and sysnonymous variants
#[287, 297): gene level features
#[297, 351): gtex gene expression data
'''


#both start and end are 1-based and inclusive.
def build_one_transcript(transcript_id):
    hhblits_path = f'{feature_dir}/{transcript_id}.99.diff0.hhm.npy'
    netsurfp2_path = f'{feature_dir}/{transcript_id}.netsurfp2.json'
    compara_path = f'{feature_dir}/{transcript_id}.compara103'
    region_path = f'{feature_dir}/{transcript_id}.aa.obs_exp.npy'

    if not os.path.exists(hhblits_path) or not os.path.exists(
            region_path) or not os.path.exists(
                netsurfp2_path) or not os.path.exists(compara_path):
        #print(hhblits_path, region_path, netsurfp2_path, compara_path)
        
        return

    compara = read_compara(compara_path)
    hhblits = np.load(hhblits_path)[:, :21]
    struc = read_netsurfp2(netsurfp2_path)
    region = read_region(region_path)

    #hhblits_path2 = f'{feature_dir}/{transcript_id}.compara.hhm.npy'
    #hhblits_path3 = f'{feature_dir}/{transcript_id}.90.diff0.hhm.npy'
    #hhblits_path4 = f'{feature_dir}/{transcript_id}.99.diff0.hhm.npy'
    #hhblits2 = np.load(hhblits_path2)[:, 1:21]
    #hhblits3 = np.load(hhblits_path3)[:, 1:21]
    #hhblits4 = np.load(hhblits_path4)[:, 1:21]

    #L = hhblits.shape[0]
    #gene = genelevel.get(transcript_id, genelevel_default)[np.newaxis, :]
    #gene = np.repeat(gene, L, axis=0)

    #print(hhblits.shape, hhblits2.shape, hhblits3.shape, hhblits4.shape,
    #      compara.shape, struc.shape, region.shape, gene.shape)
    # print(('hhblits' ,hhblits.shape))
    # print(('compara', compara.shape))
    # print(('struc', struc.shape))
    # print(('region', region.shape))
    feature = np.concatenate([hhblits, compara, struc, region], axis=-1)
    # print(feature.shape)

    output_path = f'{output_dir}/{transcript_id}.pickle'
    with open(output_path, 'wb') as fw:
        pickle.dump(feature, fw)
    #return feature


def build(name_list):
    for idx, transcript_id in enumerate(name_list):
        try:
            build_one_transcript(transcript_id)
        except:
            print(f'error {transcript_id}')
            traceback.print_exc()


def build_multi_thread(input_path, cpu):
    df = pd.read_csv(input_path, sep = '\t')
    transcript_list = list(df['transcript_id'].unique())
    #print(('eee', transcript_list))
    if cpu <= 1:
        build(transcript_list)
    else:
        num_each = int((len(transcript_list) - 1) / cpu) + 1

        pool = []
        for idx in range(cpu):
            start = idx * num_each
            end = start + num_each
            if idx == cpu - 1:
                end = len(transcript_list)
            p = Process(target=build, args=(transcript_list[start:end], ))
            pool.append(p)

        for p in pool:
            p.start()
        for p in pool:
            p.join()
#%%

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()

    build_multi_thread(args.input, args.cpu)

# %%
