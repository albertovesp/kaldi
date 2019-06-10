#!/home/hzili1/tools/anaconda3/envs/py36/bin/python

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import kaldi_io

class SPKID_Dataset(Dataset):
    """Speaker verification dataset."""
    def __init__(self, egs_file, mean=None, std=None):
        self.mean = mean
        self.std = std
        self.feat_list, self.label_list = self.read_egs_file(egs_file)

    def read_egs_file(self, egs_file):
        with open(egs_file, 'r') as fh:
            content = fh.readlines()
        feat_list, label_list = [], []
        for line in content:
            line = line.strip('\n')
            line_split = line.split()
            label_list.append(int(line_split[0].split('-')[-1]))
            feat_list.append(line_split[1])
        return feat_list, label_list

    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, idx):
        feat = kaldi_io.read_mat(self.feat_list[idx])
        assert self.mean is not None and self.std is not None
        feat = (feat - self.mean) / self.std
        return feat, self.label_list[idx]

class SPKID_Dataset_EVAL(Dataset):
    """Speaker verification dataset for evaluation."""
    def __init__(self, data_dir):
        feats_scp_file = "{}/feats.scp".format(data_dir)
        vad_scp_file = "{}/vad.scp".format(data_dir)
        self.feat_list = self.read_scp_file(feats_scp_file)
        self.vad_list = self.read_scp_file(vad_scp_file)
        assert len(self.feat_list) == len(self.vad_list)
        
    def read_scp_file(self, scp_file):
        with open(scp_file, 'r') as fh:
            content = fh.readlines()
        return content

    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, idx):
        return self.feat_list[idx], self.vad_list[idx]

class SPKDIAR_Dataset_EVAL(Dataset):
    """Speaker diarization dataset for evaluation."""
    def __init__(self, data_dir, mean=None, std=None):
        feats_scp_file = "{}/feats.scp".format(data_dir)
        self.feat_list = self.read_scp_file(feats_scp_file)
        self.mean = mean
        self.std = std
        
    def read_scp_file(self, scp_file):
        with open(scp_file, 'r') as fh:
            content = fh.readlines()
        return content

    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, idx):
        feat = kaldi_io.read_mat(self.feat_list[idx])
        assert self.mean is not None and self.std is not None
        feat = (feat - self.mean) / self.std
        return feat

def main():
    egs_file = "exp/xvector_pytorch/egs/egs.1.scp" 
    spkid_dataset = SPKID_Dataset(egs_file)
    for i in range(len(spkid_dataset)):
        feat, label = spkid_dataset[i]
        print(feat.shape)
        print(label)
        break
    return 0

def main():
    egs_file = "exp/xvector_pytorch/egs/egs.1.scp" 
    spkid_dataset = SPKID_Dataset(egs_file)
    for i in range(len(spkid_dataset)):
        feat, label = spkid_dataset[i]
        print(feat.shape)
        print(label)
        break
    return 0

if __name__ == "__main__":
    main()
