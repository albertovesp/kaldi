#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Desh Raj
# Copyright 2018  Johns Hopkins University

"""Use unigram language model for subwords to learn a variable-length encoding of the vocabulary in a text.

Reference:
Taku Kudo (2018). Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. 
Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL 2018). 
"""

from __future__ import unicode_literals

import codecs
import sys
import argparse
import sentencepiece

# hack for python2/3 compatibility
from io import open
argparse.open = open

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn unigram-LM-based word segmentation")

    parser.add_argument(
        '--input', '-i',
        help="Input text (default: standard input).")

    parser.add_argument(
        '--symbols', '-s', type=int, default=10000,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s))")
   
    parser.add_argument(
        '--output', '-o', default='model',
        help="Path to output unigram model")

    return parser

def create_train_args(infile, outfile, num_symbols):
    arg = '--input='
    arg += infile
    arg += ' --model_prefix='
    arg += outfile
    #arg += '--vocab_size='
    #arg += str(num_symbols)
    arg += " --hard_vocab_limit=false"

    return arg

def main(infile, outfile, num_symbols):
    """Learn num_symbols unigram LM subword operations from vocabulary.
    """
    train_args = create_train_args(infile, outfile, num_symbols)
    sentencepiece.SentencePieceTrainer.Train(train_args)
    
if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()

    main(args.input, args.output, args.symbols)
