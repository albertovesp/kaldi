#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Johns Hopkins University (author: Yiming Wang)
# Apache 2.0

""" This script prepares the speech commands data into kaldi format.
"""


import argparse
import os
import sys
from collections import OrderedDict
import glob

def main():
    parser = argparse.ArgumentParser(description="""Parse cost files.""")
    parser.add_argument('file1', type=str,
                        help='cost file')
    parser.add_argument('file2', type=str,
                        help='true file')
 
    args = parser.parse_args()

    with open(args.file1, 'r') as f1, open(args.file2, 'r') as f2:
        lines1, lines2 = f1.readlines(), f2.readlines()
    
    dict_ref = {}
    for line2 in lines2:
        k, v =  line2.strip().split(None, 1)
        assert k not in dict_ref
        dict_ref[k] = v

    assert len(lines1) == len(dict_ref)
    for i in range(len(lines1)):
        line1  = lines1[i].strip().split(None, 1)
        if float(line1[1]) < 500 and float(line1[1]) > 0.0 and '嗨小问' not in dict_ref[line1[0]]:
            print(line1[0])

if __name__ == "__main__":
    main()
