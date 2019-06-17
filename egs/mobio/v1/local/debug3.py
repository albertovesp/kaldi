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
                        help='wake word cost file')
    parser.add_argument('file2', type=str,
                        help='non-wake word cost file')
    parser.add_argument('file3', type=str,
                        help='non-wake word cost file')
 
    args = parser.parse_args()

    dict_ref = {}
    with open(args.file3, 'r') as f3:
        lines3 = f3.readlines()
    for line3 in lines3:
        k, v =  line3.strip().split(None, 1)
        assert k not in dict_ref
        dict_ref[k] = v

    with open(args.file1, 'r') as f1, open(args.file2, 'r') as f2:
        lines1, lines2 = f1.readlines(), f2.readlines()

    assert len(lines1) == len(lines2) and len(lines1) == len(lines3)
    for i in range(len(lines1)):
        line1, line2 = lines1[i].strip().split(None, 1), lines2[i].strip().split(None, 1)
        assert line1[0] == line2[0]
        v = dict_ref[line1[0]]
        if '嗨小问' in line1[1] and '嗨小问' not in v and '嗨小问' not in line2[1]:
            print(line1[0])

if __name__ == "__main__":
    main()
