#!/usr/bin/env python3

# Copyright      2017  Babak Rekabdar
#                2017  Hossein Hadian
#                2017  Chun Chieh Chang
#                2017  Ashish Arora
# Apache 2.0

# This script prepares lexicon for BPE. It gets the set of all words that occur in data/train/text.
# Since this lexicon is based on BPE, it replaces '|' with silence.

import argparse
import os

parser = argparse.ArgumentParser(description="""Creates the list of characters and words in lexicon""")
parser.add_argument('file', type=str, help='input file')
args = parser.parse_args()

### main ###
lex = {"<sil>":"SIL", "<unk>":"UNK"}
text_path = os.path.join('data','local',args.file)
with open(args.file, 'r', encoding='utf-8') as f:
    for line in f:
        line_vect = line.strip().split(' ')
        for word in line_vect:
            # Skip <unk>
            if (word[-1] == '>'):
                continue
            # Put SIL instead of "|". Because every "|" in the beginning of the words is for initial-space of that word
            characters = " ".join([ 'SIL' if char == '|' else char for char in word])
            lex[word] = characters
            if word == '#':
                lex[word] = "<HASH>"

with open(os.path.join('data','local','lexicon.txt'), 'w', encoding='utf-8') as fp:
    for key in sorted(lex):
        fp.write(key + " " + lex[key] + "\n")
