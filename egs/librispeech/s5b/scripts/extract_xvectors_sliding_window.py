#!/home/hzili1/tools/anaconda3/envs/py36/bin/python
# Copyright 2018 Zili Huang

# Apache 2.0

import os
import torch
import argparse
import random
from utils import extract_sliding_window
from dataprocess import SLIDING_WINDOW
from models.tdnn import tdnn_sid_xvector, tdnn_sid_xvector_1, tdnn_sid_xvector_2
import sys
import kaldi_io
import numpy as np

parser = argparse.ArgumentParser(
    description='X-vector extraction sliding window')
parser.add_argument('data_dir', type=str,
                    help='data directory')
parser.add_argument('trained_model', type=str,
                    help='path to trained model')
parser.add_argument('exp_dir', type=str,
                    help='experiment directory')
parser.add_argument('output_dir', type=str,
                    help='output directory')
parser.add_argument('--batch_size', default=1024, type=int,
                    help='mini-batch size (default: 1024)')
parser.add_argument('--arch', default='tdnn', type=str,
                    help='model architecture')
parser.add_argument('--feat_dim', default=23, type=int,
                    help='number of features for each frame')
parser.add_argument('--half_window', default=50, type=int,
                    help='half size of sliding window in frames')
parser.add_argument('--multi_gpu', default=0, type=int,
                    help='training with multiple gpus')
parser.add_argument('--embedding_type', default="a", type=str,
                    help='type of embedding (default a)')

random.seed(0)

def main():
    global args
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model
    if args.arch == 'tdnn':
        model = tdnn_sid_xvector(args.feat_dim) 
    elif args.arch == 'tdnn1': # the statistic pooling layer dimension is lower (512 vs 1500)
        model = tdnn_sid_xvector_1(args.feat_dim) 
    elif args.arch == 'tdnn2': # embedding dim 128 (for diarization)
        model = tdnn_sid_xvector_2(args.feat_dim) 
    else:
        raise ValueError("Model type not defined.")

    print(model)
    if args.multi_gpu:
        model = torch.nn.DataParallel(model)
        print("Using {} GPUs".format(torch.cuda.device_count()))
    model = model.to(device)

    if args.trained_model:
        if os.path.isfile(args.trained_model):
            print("=> loading model '{}'".format(args.trained_model))
            trained_model = torch.load(args.trained_model)
            model.load_state_dict(trained_model['state_dict'])
            print("=> loaded trained model '{}' (epoch {} subfile {})"
                  .format(args.trained_model, trained_model['epoch'], trained_model['subfile']))
            print("=> best acc {}".format(trained_model['best_acc']))
        else:
            raise ValueError(
                "=> no trained model found at '{}'".format(args.trained_model))

    mean, std = np.load("{}/mean.npy".format(args.exp_dir)), np.load("{}/std.npy".format(args.exp_dir))
    extract_dataset = SLIDING_WINDOW(args.data_dir, mean, std)
    extract_sliding_window(extract_dataset, model, device, args)
    return 0

if __name__ == "__main__":
    main()
