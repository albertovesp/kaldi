#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Desh Raj
# Copyright 2018  Johns Hopkins University

"""Use unigram language model for subwords to encode given text in the form of a subword sequence sampled from the unigram probability distribution. 

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
        description="apply unigram-LM-based word segmentation")

    parser.add_argument(
        '--input', '-i',
        help="Input text file.")

    parser.add_argument(
        '--model', '-m', default='model',
        help="Path to the unigram subword model")
   
    parser.add_argument(
        '--output', '-o', default=sys.stdout,
        help="Path to output file")

    return parser

def main(infile, model, outfile):
    """Apply unigram LM subword segmentation to given text.
    """
    sp = sentencepiece.SentencePieceProcessor()
    model_name = model + ".model"
    sp.Load(model_name)
    fout = codecs.open(outfile,'w',encoding="utf-8")
    with codecs.open(infile, encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            sw = [' '.join(sp.EncodeAsPieces(word)) for word in line]
            sw = [x.replace("\xe2\x96\x81","|") for x in sw]
            sw = ' '.join(sw)
            fout.write(sw + '\n')
    fout.close()
    
if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()
    
    main(args.input, args.model, args.output)
