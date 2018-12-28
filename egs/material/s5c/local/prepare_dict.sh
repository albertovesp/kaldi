#!/bin/bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error
echo "$0 " "$@"

data=$1
echo "$0: Preparing lexicon.."
local/prepare_lexicon.py $data

lexicon=data/local/lexicon.txt
cat $lexicon | cut -f2-  > data/local/lexicon_phns

[ ! -f $lexicon ] && echo "Lexicon $lexicon does not exist!" && exit 1;
echo $0: using lexicon $lexicon
mkdir -p data/local/dict_nosp/
cat data/train/text | cut -f 2- -d ' ' | \
  sed 's/ /\n/g' | grep . | sort -u > data/local/dict_nosp/wordlist

cp $lexicon data/local/dict_nosp/lexicon.txt
[ -f  data/local/dict_nosp/lexiconp.txt ] && rm data/local/dict_nosp/lexiconp.txt

cat data/local/dict_nosp/lexicon.txt | sed 's/\t/ /g' | \
  cut -f 2- -d ' ' | sed 's/ /\n/g' | grep . | sort -u > data/local/dict_nosp/phones.txt


echo "SIL" > data/local/dict_nosp/silence_phones.txt
cut -d' ' -f2- $lexicon | sed 's/SIL//g' | tr ' ' '\n' | sort -u | sed '/^$/d' > data/local/dict_nosp/nonsilence_phones.txt
echo "SIL" > data/local/dict_nosp/optional_silence.txt
echo "<unk>" > data/local/dict_nosp/oov.txt
echo 1 > data/local/dict_nosp/oov.int

utils/validate_dict_dir.pl data/local/dict_nosp/

