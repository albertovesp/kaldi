#!/usr/bin/env python3

# This script, prepend '|' to every words in the transcript to mark
# the beginning of the words for finding the initial-space of every word
# after decoding. It also removes all Noise tokens, i.e. those within
# '< >' except '<unk>'. If this makes the transcription empty, an <unk>
# token is added.

import sys
import io
import re

whitespace = re.compile("[ \t]+")
infile = io.TextIOWrapper(sys.stdin.buffer, encoding='latin-1')
output = io.TextIOWrapper(sys.stdout.buffer, encoding='latin-1')
for line in infile:
    words = whitespace.split(line.strip(" \t\r\n"))
    words_nosp = [e for e in words if e[0] != "<" or e == "<unk>"]
    if len(words_nosp) == 0:
        words_nosp.append("<unk>")
    output.write(' '.join([ "|"+word for word in words_nosp]) + '\n')
