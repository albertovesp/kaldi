#!/usr/bin/env python3
# coding: utf-8

# Author: Desh Raj (Johns Hopkins University) 2018

"""Use a variant of LZW compression to learn a variable-length encoding of the vocabulary in a text.
It is similar to the algorithm presented in this paper from Interspeech '05: 

https://pdfs.semanticscholar.org/dfcd/6bb8dcbcf828f8414c494fa56e96f8169a7b.pdf

Note that we learn positional tables as well but currently they are not being used for subword
segmentation.
"""


import pickle
import string
import argparse
import sys
from tqdm import tqdm

from io import open
argparse.open = open

def create_parser():

    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn LZW-based word segmentation tables")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input text (default: standard input).")

    parser.add_argument(
        '--output', '-o', type=argparse.FileType('wb'), default=None,
        metavar='PATH',
        help="Output file for LZW tables")
    parser.add_argument(
        '--symbols', '-s', type=int, default=10000,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s))")
    parser.add_argument(
        '--maxlen', '-m', type=int, default=10,
        help="Maximum subword length in table.")

    return parser


def learn_subwords_from_word(word, tab_seqlen, tab_pos, max_subword_length):
    w = ""
    pos = 0
    for i,c in enumerate(word):
        if (i == len(word) - 1):
            pos = 2
        wc = w + c
        if (len(wc) > max_subword_length):
            wc = c
        
        if wc in tab_seqlen[len(wc)-1]:
            w = wc
            tab_seqlen[len(wc)-1][wc] += 1
        else:
            tab_seqlen[len(wc)-1][wc] = 1
            w = c
        if wc in tab_pos[pos]:
            w = wc
            tab_pos[pos][wc] += 1
            i -= 1
        else:
            tab_pos[pos][wc] = 1
            w = c
            pos = min(i,1)


def prune_tables(tab_seqlen, tab_pos, num_symbols):
    if (len(tab_seqlen) < 2):
        return
    combined_dict = {}
    for d in tab_seqlen[1:]:
        combined_dict.update(d)
    
    if (num_symbols > len(combined_dict)+len(tab_seqlen[0])):
        return

    ranked_subwords = [pair[0] for pair in sorted(combined_dict.items(), reverse=True, key=lambda item: item[1])]
    selected_subwords = ranked_subwords[:num_symbols]
    discarded_subwords = ranked_subwords[num_symbols+1:]
    for word in discarded_subwords:
        tab_seqlen[len(word)-1].pop(word)
        for i in range(len(tab_pos)):
            tab_pos[i].pop(word, None)


def rank_words_by_count(d):
    ranked_words = [pair[0] for pair in sorted(d.items(), reverse=True, key=lambda item: item[1])]
    for i,word in enumerate(ranked_words):
        d[word] = (d[word],i+1)


def learn_lzw_subwords(infile, outfile, max_subword_length=6, num_symbols=-1):
    tab_seqlen = [{} for _ in range(max_subword_length)]
    tab_pos = [{} for _ in range(3)]
    
    # table = str.maketrans('', '', string.punctuation)
    sys.stderr.write("\nLearning LZW tables")
    
    lines = infile.readlines()
    for line in tqdm(lines):
        words = line.strip().split(' ')
        # stripped = [w.translate(table) for w in words]
        for word in words:
            if word.strip():
                learn_subwords_from_word(word, tab_seqlen, tab_pos, max_subword_length)
        
    if (num_symbols != -1):
        prune_tables(tab_seqlen, tab_pos, num_symbols)
        
    for d in tab_seqlen:
        rank_words_by_count(d)
    for d in tab_pos:
        rank_words_by_count(d)
        
    infile.close()
    
    vocab = {'tab_seqlen':tab_seqlen, 'tab_pos':tab_pos}
    pickle.dump(vocab, outfile)

    sys.stderr.write("\nStatistics of subwords learnt:\n")
    for i,tab in enumerate(tab_seqlen):
        sys.stderr.write('Subwords of length {0}: {1}\n'.format(i+1,len(tab)))
           
    return


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()

    learn_lzw_subwords(args.input, args.output, args.maxlen, args.symbols)






