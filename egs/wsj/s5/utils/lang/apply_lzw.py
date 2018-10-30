#!/usr/bin/env python3
# coding: utf-8

# Author: Desh Raj (Johns Hopkins University) 2018

"""This script applies the learnt LZW compression tables to obtain a segmentation of the text.
Top k segmentations are retrieved based on scores calculated as follows:

Score of a segmentation = sum of all subword scores
where score of subword = weight x relative rank of subword in its table
Here, weight = (1/length of subword)^(maxlen) 

This score empirically gives subword lengths which correspond closely with the distribution of 
syllable lengths in English. It is a variation of the scoring scheme proposed in this paper:

https://pdfs.semanticscholar.org/dfcd/6bb8dcbcf828f8414c494fa56e96f8169a7b.pdf

One segmentation is then sampled from the multinomial distribution of the top-k based
on the probability distribution:
p_k = (1/s_k)^alpha / Z

where s_k = score of candidate k, alpha = smoothing hyperparameter, 
and Z = normalization constant
"""

import numpy as np
import pickle
from scipy.special import expit
# import string
import copy
import sys
import argparse
from tqdm import tqdm

argparse.open = open

def create_parser():

    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Apply LZW-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input text (default: standard input).")

    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=None,
        metavar='PATH',
        help="Output file for storing segmented text")

    parser.add_argument(
        '--vocab', '-v', type=argparse.FileType('rb'), default=None,
        metavar='PATH',
        help="Vocab file containing LZW tables")

    parser.add_argument(
        '--top-k', '-k', type=int, default=5,
        help="Top k segmentations to sample from. For Viterbi segmentation, set k as 1.")

    parser.add_argument(
        '--alpha', '-a', type=float, default=0.1,
        help="Smoothing hyperparameter")

    return parser


def compute_segment_scores(word, tab_seqlen, tab_pos, scores):
    if (len(word) == 0):
        return ([])
    max_subword_length = len(tab_seqlen)
    seg_scores = []

    for i in range(max_subword_length):
        if(i < len(word)):
            subword = word[:i+1]
            if subword in tab_seqlen[i]:
                other_scores = []
                subword_score = float(tab_seqlen[i][subword][1]/(((i+1)**max_subword_length)*len(tab_seqlen[i])))
                if (word[i+1:] in scores):
                    other_scores = copy.deepcopy(scores[word[i+1:]])
                else:
                    other_scores = copy.deepcopy(compute_segment_scores(word[i+1:], tab_seqlen, tab_pos, scores))
                if (len(other_scores) == 0):
                    seg_scores.append(([subword],subword_score))
                else:
                    for j,segment in enumerate(other_scores):
                        other_scores[j] = ([subword]+segment[0],subword_score+segment[1])
                    seg_scores += other_scores
    seg_scores = sorted(seg_scores, key=lambda item: item[1])
    scores[word] = seg_scores
    return seg_scores


def segment_word(word, segment_dict, scores, tab_seqlen, tab_pos, k, alpha):
    if word not in segment_dict:
        if (len(word)>16):
            return (segment_long_word(word, segment_dict, scores, tab_seqlen, tab_pos, k, alpha))
        seg_scores = compute_segment_scores(word, tab_seqlen, tab_pos, scores)
        k = min(k,len(seg_scores))
        segment_dict[word] = seg_scores[:k]
    sampled_index = sample_segment([i[1] for i in segment_dict[word]], alpha)
    return (' '.join(segment_dict[word][sampled_index][0]))


def segment_long_word(word, segment_dict, scores, tab_seqlen, tab_pos, k, alpha):
    segment1 = segment_word(word[:int(len(word)/2)], segment_dict, scores, tab_seqlen, tab_pos, k, alpha)
    segment2 = segment_word(word[int(len(word)/2):], segment_dict, scores, tab_seqlen, tab_pos, k, alpha)
    return (segment1 + ' ' + segment2)


def sample_segment(scores, alpha):
    scores = np.array([(1/i) for i in scores])
    scores = np.power(expit(scores),alpha)
    scores = scores/sum(scores)
    sampled_index = np.argmax(np.random.multinomial(1,scores))
    return sampled_index


def apply_lzw_segmentation(infile, outfile, vocab_file, k=-1, alpha=0.1):
    fr = infile
    fw = outfile
    
    data = pickle.load(vocab_file)
    tab_seqlen = data['tab_seqlen']
    tab_pos = data['tab_pos']
    
    # table = str.maketrans('', '', string.punctuation)
    segment_dict = {}
    subword_scores = {}
    
    lines = fr.readlines()
    sys.stderr.write('Segmenting {0} lines'.format(len(lines)))

    for line in tqdm(lines):
        segmented_line = ['']
        words = line.strip().split(' ')
        # stripped = [w.translate(table) for w in words]
        for word in words:
            if word.strip():
                subwords = segment_word(word, segment_dict, subword_scores, tab_seqlen, tab_pos, k, alpha)
                segmented_line.append(subwords)
        segmented_line = ' |'.join(segmented_line)
        fw.write(segmented_line+'\n')
    
    fr.close()
    fw.close()
    
    return


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()

    apply_lzw_segmentation(args.input, args.output, args.vocab, args.top_k, args.alpha)

