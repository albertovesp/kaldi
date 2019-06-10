#!/home/hzili1/tools/anaconda3/envs/py27/bin/python

import numpy as np
import sys
import torch

def main():
    model_filename = sys.argv[1]
    print("=> loading model '{}'".format(model_filename))
    model = torch.load(model_filename)
    epoch = model['epoch']
    subfile = model['subfile']
    best_acc = model['best_acc']
    print("=> loaded model '{}' (epoch {} subfile {})"
        .format(model_filename, epoch, subfile))
    print("best acc {}".format(best_acc))
    return 0

if __name__ == "__main__":
    main()
