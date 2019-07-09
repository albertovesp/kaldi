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
                    help='keep only this many words by frequency')

def get_int_labels(label_file, top_n):
  utt2words = {}
  vocab = []
  with open(label_file, 'r') as fh:
    samples = fh.readlines()
  for sample in samples:
    utt, label = sample.strip('\n').split(' ', 1)
    words = list(set(re.sub("[^\w]", " ",  label.lower()).split()))
    utt2words[utt] = words
    vocab += words
  counts = Counter(vocab)
  top_n_words = [word for word, count in counts.most_common(top_n)]
  for utt in utt2words:
    for word in utt2words[utt]:
      if word not in top_n_words:
        utt2words[utt].remove(word)
  utt2labels = {}
  top_n_counts = {}
  for word in top_n_words:
    top_n_counts[word] = vocab.count(word)
    for utt in utt2words:
      label = int(word in utt2words[utt])
      if utt in utt2labels:
        utt2labels[utt].append(label)
      else:
        utt2labels[utt] = [label]
  return utt2labels, top_n_words, top_n_counts

def main():
  global args
  args = parser.parse_args()
  print(args)

  # set random seed
  random.seed(args.seed)
  np.random.seed(args.seed)
  #torch.manual_seed(args.seed)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
  label_list, top_n_words, top_n_counts = get_int_labels(args.label_file, args.top_n)
  
  fout = open(args.output_file, 'a')
  fout.write('System: {}\n'.format(args.vector_scp_file))
  fout.write('Word \t Count \t Accuracy\n')

  words_correct = []
  for k,word in enumerate(top_n_words):
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
    y_true = []
    y_pred = []
    with torch.no_grad():
      words_correct_k = []
      for data in test_loader:
        inputs, labels = data
        outputs = torch.squeeze(model(inputs))
        _, predicted = torch.max(outputs.data, 1)
        y_true += labels.tolist()
        y_pred += predicted.tolist()  
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        words_correct_k += (predicted == labels).tolist()
    if (k==0):
      words_correct = words_correct_k
    else:
      words_correct = [sum(x) for x in zip(words_correct, words_correct_k)]
    print('Accuracy: %.2f %%' % (100 * correct / total))
    report = classification_report(y_true, y_pred)

    fout.write('{}\t{:d}\t{:f}\t{}\n'.format(word, top_n_counts[word], 100*correct / total,
      precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)))
 
  total_words_correct = sum(words_correct)
  total_words = len(words_correct) * args.top_n
  accuracy_correct = 100 * float(total_words_correct) / total_words
  num_all_correct = words_correct.count(args.top_n)
  accuracy_all_correct = 100 * float(num_all_correct) / len(words_correct)
  fout.write('Average %% of words correct: %.2f %%\n' % (accuracy_correct))
  fout.write('%% of utterances which got all words correct: %.2f %%\n' % (accuracy_all_correct))
  for x in Counter(words_correct).most_common():
    fout.write('{} '.format(x))
  fout.write('\n--------------------------------\n')

if __name__ == "__main__":
  main()
