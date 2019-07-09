#!/home/draj/anaconda3/envs/espnet/bin/python

# Copyright  2019 Johns Hopkins University (Author: Desh Raj)

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kaldi_io

def read_as_dict(file):
  utt2feat = {}
  with open(file, 'r') as fh:
    content = fh.readlines()
  for line in content:
    line = line.strip('\n')
    utt2feat[line.split()[0]] = line.split()[1]
  return utt2feat

class SPKID_Dataset(Dataset):
  # Loads i-vectors/x-vectors and labels for the data
  def __init__(self, vector_scp, labels):
    self.feat_list, self.label_list = self.read_scp_labels(vector_scp, labels)

  def read_scp_labels(self, vector_scp, utt2label):
    utt2scp = read_as_dict(vector_scp)
    feat_list = []
    label_list = []
    for utt in utt2scp:
      if utt in utt2label:
        feat_list.append(utt2scp[utt])
        label_list.append(utt2label[utt])
    return feat_list, label_list

  def __len__(self):
    return len(self.feat_list)

  def __getitem__(self, idx):
    feat = kaldi_io.read_mat(self.feat_list[idx])
    return feat, self.label_list[idx]


class Classifier(nn.Module):
  # Trains a linear classifier on the speaker embeddings
  def __init__(self, feat_dim, num_classes):
    super(Classifier, self).__init__()
    self.fc1 = nn.Linear(feat_dim, 500)
    self.fc2 = nn.Linear(500, num_classes)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    y = self.relu(x)
    return y
        
class Regressor(nn.Module):
  # Trains a linear classifier on the speaker embeddings
  def __init__(self, feat_dim):
    super(Regressor, self).__init__()
    self.fc1 = nn.Linear(feat_dim, 500)
    self.fc2 = nn.Linear(500, 1)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    y = self.relu(x).double()
    return y
