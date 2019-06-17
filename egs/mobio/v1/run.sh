#!/bin/bash

stage=0


. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -euo pipefail

if [ $stage -le 0 ]; then
  local/mobvoi_data_download.sh
  echo "$0: Extracted all datasets into data/download/"
fi

if [ $stage -le 1 ]; then
  echo "$0: Splitting datasets..."
  local/split_datasets.sh
  echo "$0: text and utt2spk have been generated in data/{train|dev|eval}."
fi
    
if [ $stage -le 2 ]; then
  echo "$0: Preparing wav.scp..."
  local/prepare_wav.py data
  echo "wav.scp has been generated in data/{train|dev|eval}."
fi

if [ $stage -le 3 ]; then
  echo "$0: Extracting MFCC..."
  for folder in train dev eval; do
    dir=data/$folder
    utils/fix_data_dir.sh $dir
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 16 $dir
    steps/compute_cmvn_stats.sh $dir
    utils/fix_data_dir.sh $dir
    utils/data/get_utt2dur.sh $dir
    utils/validate_data_dir.sh $dir
  done
fi

if [ $stage -le 4 ]; then
  echo "$0: extracting hires MFCC features for the dev/eval data..."
  for datadir in train dev eval; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done
  for datadir in train dev eval; do
    steps/make_mfcc.sh --nj 30 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires || exit 1;
    utils/fix_data_dir.sh data/${datadir}_hires
  done
fi


