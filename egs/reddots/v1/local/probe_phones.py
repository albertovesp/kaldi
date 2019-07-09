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
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
import re
from g2p_en import G2p

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
parser.add_argument('--output_file', default='RESULTS', type=str,
                    help='file to store result')
parser.add_argument('--top_n', default=50, type=int,
                    help='keep only this many phones by frequency')

def get_int_labels(label_file, top_n):
  utt2phones = {}
  vocab = []
  g2p = G2p()
  consonants = ['B', 'CH', 'D', 'DH', 'ER0', 'F', 'G', 'JH', 'K', 'L',
    'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
  with open(label_file, 'r') as fh:
    samples = fh.readlines()
  for sample in samples:
    utt, label = sample.strip('\n').split(' ', 1)
    label = re.sub("[^\w]", " ", label.lower())
    phones_all = [x for x in g2p(label) if x in consonants]
    phones = list(filter(lambda a: phones_all.count(a) < 3, phones_all))
    utt2phones[utt] = phones
    vocab += phones
  counts = Counter(vocab)
  top_n_phones = [phone for phone, count in counts.most_common(top_n)]
  for utt in utt2phones:
    for phone in utt2phones[utt]:
      if phone not in top_n_phones:
        utt2phones[utt].remove(phone)
  utt2labels = {}
  top_n_counts = {}
  for phone in top_n_phones:
    top_n_counts[phone] = vocab.count(phone)
    for utt in utt2phones:
      label = int(phone in utt2phones[utt])
      if utt in utt2labels:
        utt2labels[utt].append(label)
      else:
        utt2labels[utt] = [label]
  return utt2labels, top_n_phones, top_n_counts

def main():
  global args
  args = parser.parse_args()
  print(args)

  # set random seed
  random.seed(args.seed)
  np.random.seed(args.seed)
  #torch.manual_seed(args.seed)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
  label_list, top_n_phones, top_n_counts = get_int_labels(args.label_file, args.top_n)
  
  fout = open(args.output_file, 'a')
  fout.write('System: {}\n'.format(args.vector_scp_file))
  fout.write('Word \t Count \t Accuracy\n')

  phones_correct = []
  for k, phone in enumerate(top_n_phones):
    labels = {}
    for utt in label_list:
      labels[utt] = label_list[utt][k]
    dataset = SPKID_Dataset(args.vector_scp_file, labels)
    dataset_size = len(dataset)
    
    if (k == 0):
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

    model = Classifier(args.feat_dim, 2)
    print(model)

    if (args.multi_gpu):
      model = torch.nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss()
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
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
      phones_correct_k = []
      for data in test_loader:
        inputs, labels = data
        outputs = torch.squeeze(model(inputs))
        _, predicted = torch.max(outputs.data, 1)
        y_true += labels.tolist()
        y_pred += predicted.tolist()  
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        phones_correct_k += (predicted == labels).tolist()
    if (k==0):
      phones_correct = phones_correct_k
    else:
      phones_correct = [sum(x) for x in zip(phones_correct, phones_correct_k)]
    print('Accuracy: %.2f %%' % (100 * correct / total))
    report = classification_report(y_true, y_pred)

    fout.write('{}\t{:d}\t{:f}\t{}\n'.format(phone, top_n_counts[phone], 100*correct / total,
      precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)))
 
  total_phones_correct = sum(phones_correct)
  total_phones = len(phones_correct) * args.top_n
  accuracy_correct = 100 * float(total_phones_correct) / total_phones
  num_all_correct = phones_correct.count(args.top_n)
  accuracy_all_correct = 100 * float(num_all_correct) / len(phones_correct)
  fout.write('Average %% of phones correct: %.2f %%\n' % (accuracy_correct))
  fout.write('%% of utterances which got all phones correct: %.2f %%\n' % (accuracy_all_correct))
  for x in Counter(phones_correct).most_common():
    fout.write('{} '.format(x))
  fout.write('\n--------------------------------\n')

if __name__ == "__main__":
  main()
