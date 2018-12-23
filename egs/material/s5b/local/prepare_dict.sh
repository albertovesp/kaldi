#!/bin/bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error
echo "$0 " "$@"

if [ $# -ne 1 ] ; then
  echo "Invalid number of script parameters. "
  echo "  $0 <path-to-material-corpus>"
  echo "e.g."
  echo "  $0 /export/corpora5/MATERIAL/IARPA_MATERIAL_BASE-1A-BUILD_v1.0/"
  exit
fi
data=$1
 
mkdir -p data/local
phonemic_lexicon=$data/conversational/reference_materials/lexicon.txt
cat $phonemic_lexicon | awk '{print $1}' > data/local/lexicon_words
local/prepare_lexicon.py data/local/lexicon_words

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


echo "<sil>" > data/local/dict_nosp/silence_phones.txt
grep -v "^<.*>$" data/local/dict_nosp/phones.txt  > data/local/dict_nosp/nonsilence_phones.txt
echo "<sil>" > data/local/dict_nosp/optional_silence.txt
echo "<unk>" > data/local/dict_nosp/oov.txt



utils/validate_dict_dir.pl data/local/dict_nosp/

