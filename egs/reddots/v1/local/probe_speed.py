#!/home/draj/anaconda3/envs/espnet/bin/python

# Copyright  2019 Johns Hopkins University (Author: Desh Raj)

import torch
import torch.nn as nn
from probe_utils import SPKID_Dataset, Classifier
import numpy as np
import random
import torch.optim as optim
import socket
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn import preprocessing
from sklearn.metrics import classification_report
from collections import Counter

print(socket.gethostname())

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(
    description='Speaker ID classification probing task')
parser.add_argument('vector_scp_file', type=str,
                    help='path to scp file for speaker embeddings')
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
parser.add_argument('--output_file', default='RESULTS', type=str,
                    help='file to store result')

def get_int_labels(scp_file):
  utts = []
  labels = []
  with open(scp_file, 'r') as fh:
    samples = fh.readlines()
  for sample in samples:
    utt, scp = sample.strip('\n').split()
    utts.append(utt)
    labels.append(utt[2:5])
  le = preprocessing.LabelEncoder()
  int_labels = le.fit_transform(labels)
  utt2label = dict(zip(utts, int_labels))
  return utt2label, len(le.classes_), le


def main():
  global args
  args = parser.parse_args()
  print(args)

  # set random seed
  random.seed(args.seed)
  np.random.seed(args.seed)
  #torch.manual_seed(args.seed)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  labels, num_classes, le = get_int_labels(args.vector_scp_file)
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

  model = Classifier(args.feat_dim, num_classes)
  print(model)

  if (args.multi_gpu):
    model = torch.nn.DataParallel(model)
  
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

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
  correct = 0
  total = 0
  with torch.no_grad():
    for data in test_loader:
      inputs, labels = data
      outputs = torch.squeeze(model(inputs))
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  print('Accuracy: %.2f %%' % (100 * correct / total))
  report = classification_report(labels, predicted)

  fout = open(args.output_file, 'a')
  fout.write('System: {}\n'.format(args.vector_scp_file))
  fout.write('Train/test split: {:f}/{:f}\n'.format(1-args.test_split,args.test_split))
  fout.write('Epochs: {:d}\n'.format(args.num_epochs))
  fout.write('Final training loss: {:f}\n'.format(running_loss))
  fout.write('Test accuracy: {:f}\n'.format(100*correct / total))
  fout.write('{}\n'.format(report))
  fout.write('{}\n'.format(le.inverse_transform([0,1,2])))
  fout.write('--------------------------------\n')

if __name__ == "__main__":
  main()
