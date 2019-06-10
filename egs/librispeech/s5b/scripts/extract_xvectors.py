#!/home/hzili1/tools/anaconda3/envs/py36/bin/python
# Copyright 2018 Zili Huang

# Apache 2.0

import os
import torch
import argparse
import random
from utils import extract
from dataprocess import SPKID_Dataset, SPKID_Dataset_EVAL
from models.tdnn import tdnn_sid_xvector, tdnn_sid_xvector_1, tdnn_sid_xvector_2
import sys
import kaldi_io
import numpy as np

parser = argparse.ArgumentParser(
    description='Pytorch implementation of x-vector extraction')
parser.add_argument('data_dir', type=str,
                    help='data directory')
parser.add_argument('trained_model', type=str,
                    help='path to trained model')
parser.add_argument('output_dir', type=str,
                    help='output directory')
parser.add_argument('--batch_size', default=1024, type=int,
                    help='mini-batch size (default: 1024)')
parser.add_argument('--arch', default='tdnn', type=str,
                    help='model architecture')
parser.add_argument('--feat_dim', default=23, type=int,
                    help='number of features for each frame')
parser.add_argument('--multi_gpu', default=0, type=int,
                    help='training with multiple gpus')
parser.add_argument('--min_chunk_size', default=25, type=int,
                    help='minimum chunk size')
parser.add_argument('--max_chunk_size', default=10000, type=int,
                    help='maximum chunk size')
parser.add_argument('--drop_last', default=0, type=int,
                    help='drop the last chunk')
parser.add_argument('--train_egs_dir', default="exp/xvector_pytorch/egs", type=str,
                    help='path of training egs (load mean and std)')
parser.add_argument('--embedding_type', default="a", type=str,
                    help='type of embedding (default a)')
parser.add_argument('--num_workers', default=1, type=int,
                    help='number of workers for data loading (default: 1)')

random.seed(0)

def main():
    global args
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model
    if args.arch == 'tdnn':
        model = tdnn_sid_xvector(args.feat_dim) 
    elif args.arch == 'tdnn1':
        model = tdnn_sid_xvector_1(args.feat_dim) 
    elif args.arch == 'tdnn2':
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

    extract_dataset = SPKID_Dataset_EVAL(args.data_dir)
    extractloader = torch.utils.data.DataLoader(extract_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    extract(extractloader, model, device, args)
    return 0

if __name__ == "__main__":
    main()
