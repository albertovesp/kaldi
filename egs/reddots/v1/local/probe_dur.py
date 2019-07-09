#!/home/draj/anaconda3/envs/espnet/bin/python

# Copyright  2019 Johns Hopkins University (Author: Desh Raj)

import torch
import torch.nn as nn
from probe_utils import SPKID_Dataset, Regressor
import numpy as np
import random
import torch.optim as optim
import socket
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn import preprocessing
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.metrics import mean_squared_error
from math import sqrt

print(socket.gethostname())

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(
    description='Speaker ID classification probing task')
parser.add_argument('vector_scp_file', type=str,
                    help='path to scp file for speaker embeddings')
parser.add_argument('label_file', type=str,
                    help='path to file containing labels')
parser.add_argument('--seed', default=7, type=int,
                    help='random seed')
parser.add_argument('--test_split', default=0.1, type=float,
                    help='fraction of data to use for test')
parser.add_argument('--feat_dim', default=100, type=int,
                    help='dimensionality of the speaker embeddings')
parser.add_argument('--multi_gpu', default=0, type=int,
                    help='training with multiple gpus')
parser.add_argument('--num_epochs', default=10, type=int,
                    help='number of epochs to train')
parser.add_argument('--num_buckets', default=5, type=int,
                    help='number of buckets of utterance length')
parser.add_argument('--output_file', default='RESULTS', type=str,
                    help='file to store result')
parser.add_argument('--np_array_file', default='ARRAY', type=str,
                    help='file to store np arrays of true and predicted labels')

def get_labels(label_file):
  utts = []
  labels = []
  with open(label_file, 'r') as fh:
    samples = fh.readlines()
  for sample in samples:
    utt, label = sample.strip('\n').split()
    utts.append(utt)
    labels.append(float(label))
  
  utt2label = dict(zip(utts, labels))
  return utt2label


def main():
  global args
  args = parser.parse_args()
  print(args)

  # set random seed
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  labels  = get_labels(args.label_file)
  #print(Counter(labels.values()).most_common())
  dataset = SPKID_Dataset(args.vector_scp_file, labels)
  dataset_size = len(dataset)
  indices = list(range(dataset_size))
  np.random.shuffle(indices)
  split = int(np.floor(args.test_split * dataset_size))
  train_indices, test_indices = indices[split:], indices[:split]

  train_sampler = SubsetRandomSampler(train_indices)
  test_sampler = SubsetRandomSampler(test_indices)
  
  train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, 
                                             sampler=train_sampler)
  test_loader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                            sampler=test_sampler)

  model = Regressor(args.feat_dim)
  print(model)

  if (args.multi_gpu):
    model = torch.nn.DataParallel(model)
  
  criterion = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

  for epoch in range(args.num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = torch.squeeze(model(inputs))
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (i+1)))

  print('Finished Training')
  
  # Test on the test data
  y_true = []
  y_pred = []
  with torch.no_grad():
    for data in test_loader:
      inputs, labels = data
      outputs = torch.squeeze(model(inputs))
      y_true += labels.tolist()
      y_pred += outputs.tolist()
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  RMSE = sqrt(mean_squared_error(y_true, y_pred))
  print('RMSE: %.2f' % RMSE)
  np.savez(args.np_array_file, y_true=y_true, y_pred=y_pred)
  fout = open(args.output_file, 'a')
  fout.write('System: {}\n'.format(args.vector_scp_file))
  fout.write('Train/test split: {:f}/{:f}\n'.format(1-args.test_split,args.test_split))
  fout.write('Epochs: {:d}\n'.format(args.num_epochs))
  fout.write('Final training loss: {:f}\n'.format(running_loss))
  fout.write('Test RMSE: {:f}\n'.format(RMSE))
  fout.write('--------------------------------\n')

if __name__ == "__main__":
  main()
