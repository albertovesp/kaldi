#!/usr/bin/env bash
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe, Yenda Trmal)
# Apache 2.0

# Begin configuration section.
cleanup=true
session=S02  # S02/S09/S01/S21

# End configuration section
. ./utils/parse_options.sh  # accept options.. you can run this run.sh with the

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 3 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <audio-dir> <json-transcript-dir> <output-dir>"
  echo -e >&2 "eg:\n  $0 /corpora/chime5/audio/train /corpora/chime5/transcriptions/train data/train"
  exit 1
fi

set -e -o pipefail

adir=$1
jdir=$2
dir=$3

json_count=$(find -L $jdir -name "*.json" | wc -l)
wav_count=$(find -L $adir -name "*.wav" | wc -l)

if [ "$json_count" -eq 0 ]; then
  echo >&2 "We expect that the directory $jdir will contain json files."
  echo >&2 "That implies you have supplied a wrong path to the data."
  exit 1
fi
if [ "$wav_count" -eq 0 ]; then
  echo >&2 "We expect that the directory $adir will contain wav files."
  echo >&2 "That implies you have supplied a wrong path to the data."
  exit 1
fi

echo "$0: Converting transcription to text"

mkdir -p $dir

./local/json2text.py --mictype ref $jdir/${session}.json |\
  sed -e "s/\[inaudible[- 0-9]*\]/[inaudible]/g" |\
  sed -e 's/ - / /g' |\
  sed -e 's/mm-/mm/g' > $dir/text.orig

echo "$0: Creating datadir $dir"

# Fixed reference array
# first get a text, which will be used to extract reference arrays
perl -ne 's/-/.ENH-/;print;' $dir/text.orig | sort > $dir/text

find -L $adir | grep "\.wav" | grep "$session" | sort > $dir/wav.flist
# following command provide the argument for grep to extract only reference arrays
#grep `cut -f 1 -d"-" $dir/text | awk -F"_" '{print $2 "_" $3}' | sed -e "s/\.ENH//" | sort | uniq | sed -e "s/^/ -e /" | tr "\n" " "` $dir/wav.flist > $dir/wav.flist2
paste -d" " \
<(awk -F "/" '{print $NF}' $dir/wav.flist | sed -e "s/\.wav/.ENH/") \
$dir/wav.flist | sort > $dir/wav.scp

$cleanup && rm -f $dir/text.* $dir/wav.scp.* $dir/wav.flist

# Prepare 'segments', 'utt2spk', 'spk2utt'
cut -d" " -f 1 $dir/text | \
  awk -F"-" '{printf("%s %s %08.2f %08.2f\n", $0, $1, $2/100.0, $3/100.0)}' |\
  sed -e "s/_[A-Z]*\././2" |\
  sed -e "s/ P.._/ /" > $dir/segments

cut -f 1 -d ' ' $dir/segments | \
  perl -ne 'chomp;$utt=$_;s/_.*//;print "$utt $_\n";' > $dir/utt2spk

utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt

# For scoring the final system, we need the original utt2spk
# and text file. So we keep them with the extension .bak here
# so that they don't affect the validate_data_dir steps in
# the intermediate steps.
for file in text utt2spk spk2utt segments; do
  mv $dir/$file $dir/$file.bak
done

# Prepare pseudo utt2spk.
awk '{print $1, $1}' $dir/wav.scp > $dir/utt2spk
utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt

exit
