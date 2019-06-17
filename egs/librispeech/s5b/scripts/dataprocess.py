#!/home/hzili1/tools/anaconda3/envs/py36/bin/python

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import kaldi_io

class SPKID_Dataset(Dataset):
    """Speaker verification dataset."""
    def __init__(self, egs_file, mean, std, utt2feat):
        self.mean, self.std, self.utt2feat = mean, std, utt2feat
        self.feat_list, self.label_list, self.start_frame_list, self.duration = self.read_egs_file(egs_file)

    def read_egs_file(self, egs_file):
        with open(egs_file, 'r') as fh:
            content = fh.readlines()
        feat_list, label_list, start_frame_list, duration_list = [], [], [], []
        for line in content:
            line = line.strip('\n')
            egs_info = line.split()[0]
            egs_info_split = egs_info.split('-')
            uttname, start_frame, duration, spk_idx = '-'.join(egs_info_split[:-3]), int(egs_info_split[-3]), int(egs_info_split[-2]), int(egs_info_split[-1])

            feat_list.append(self.utt2feat[uttname])
            label_list.append(spk_idx)
            start_frame_list.append(start_frame)
            duration_list.append(duration)
        assert np.max(duration_list) == np.min(duration_list)
        duration = np.max(duration_list)
        return feat_list, label_list, start_frame_list, duration

    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, idx):
        feat = kaldi_io.read_mat(self.feat_list[idx])[self.start_frame_list[idx] : self.start_frame_list[idx] + self.duration, :]
        assert self.mean is not None and self.std is not None
        feat = (feat - self.mean) / self.std
        return feat, self.label_list[idx]
 
class SPKID_Dataset_MTL(Dataset):
    """Speaker verification dataset with multi-task (speech enhancement)."""
    def __init__(self, egs_file, mean, std, utt2feat, utt2clean):
        self.mean, self.std, self.utt2feat, self.utt2clean = mean, std, utt2feat, utt2clean
        self.feat_list, self.label_list, self.clean_feat_list, self.start_frame_list, self.duration = self.read_egs_file(egs_file)

    def read_egs_file(self, egs_file):
        with open(egs_file, 'r') as fh:
            content = fh.readlines()
        feat_list, label_list, clean_feat_list, start_frame_list, duration_list = [], [], [], [], []
        for line in zip(content):
            line = line.strip('\n')
            egs_info = line.split()[0]
            egs_info_split = egs_info.split('-')
            uttname, start_frame, duration, spk_idx = '-'.join(egs_info_split[:-3]), int(egs_info_split[-3]), int(egs_info_split[-2]), int(egs_info_split[-1])

            feat_list.append(self.utt2feat[uttname])
            clean_feat_list.append(self.utt2clean[uttname])
            label_list.append(spk_idx)
            start_frame_list.append(start_frame)
            duration_list.append(duration)
        assert np.max(duration_list) == np.min(duration_list)
        duration = np.max(duration_list)
        return feat_list, label_list, clean_feat_list, start_frame_list, duration

    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, idx):
        feat = kaldi_io.read_mat(self.feat_list[idx])[self.start_frame_list[idx] : self.start_frame_list[idx] + self.duration, :]
        clean_feat = kaldi_io.read_mat(self.clean_feat_list[idx])[self.start_frame_list[idx] : self.start_frame_list[idx] + self.duration, :]
        assert self.mean is not None and self.std is not None
        feat = (feat - self.mean) / self.std
        return feat, self.label_list[idx], clean_feat

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

class SLIDING_WINDOW(Dataset):
    def __init__(self, data_dir, mean=None, std=None):
        feats_scp_file = "{}/feats.scp".format(data_dir)
        self.feat_list, self.uttlist = self.read_scp_file(feats_scp_file)
        self.mean, self.std = mean, std
        
    def read_scp_file(self, scp_file):
        with open(scp_file, 'r') as fh:
            content = fh.readlines()
        featlist, uttlist = [], []
        for line in content:
            line  = line.strip('\n')
            uttname = line.split()[0]
            uttlist.append(uttname)
            featlist.append(line.split()[1])
        return featlist, uttlist

    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, idx):
        feat = kaldi_io.read_mat(self.feat_list[idx])
        assert self.mean is not None and self.std is not None
        feat = (feat - self.mean) / self.std
        return self.uttlist[idx], feat

def main():
    egs_file = "exp/xvector_nnet_1a/egs/egs.1.scp" 
    #mean, std = np.load("/export/b02/zili/sid/sre16_pytorch_v1/exp/xvector_pytorch/egs/mean.npy"), np.load("/export/b02/zili/sid/sre16_pytorch_v1/exp/xvector_pytorch/egs/std.npy")
    mean, std = np.zeros((23, )), np.ones((23,))
    utt2feat = {}
    with open("data/swbd_sre_combined_no_sil/feats.scp", 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        utt, feat = line.split()
        utt2feat[utt] = feat

    spkid_dataset = SPKID_Dataset(egs_file, mean, std, utt2feat)
    for i in range(len(spkid_dataset)):
        feat, label = spkid_dataset[i]
        print(feat.shape)
        print(label)
        print(feat)
        break
    return 0

if __name__ == "__main__":
    main()
