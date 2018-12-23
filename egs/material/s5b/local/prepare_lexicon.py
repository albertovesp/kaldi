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
lex = {"<unk>":"<sil>"}
text_path = os.path.join('data','local',args.file)
with open(args.file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        lex[line] = ' '.join(list(line))

with open(os.path.join('data','local','lexicon.txt'), 'w', encoding='utf-8') as fp:
    for key in sorted(lex):
        fp.write(key + " " + lex[key] + "\n")
