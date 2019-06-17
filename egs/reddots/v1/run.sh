#!/bin/bash

stage=0
db_dir=/export/corpora/ASTAR/RedDots
data_dir=data

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -euo pipefail

if [ $stage -le 0 ]; then
  mkdir -p $data_dir
  local/prepare_data.py $db_dir $data_dir
  filter_scp.pl $data_dir/utt2spk $data_dir/text > $data_dir/utt2txt
fi

if [ $stage -le 1 ]; then
  echo "$0: Extracting MFCC features for all data..."
  utils/fix_data_dir.sh $data_dir
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 16 $data_dir
  steps/compute_cmvn_stats.sh $data_dir
  utils/fix_data_dir.sh $data_dir
  utils/data/get_utt2dur.sh $data_dir
  utils/validate_data_dir.sh $data_dir
fi

if [ $stage -le 2 ]; then
  echo "$0: extracting hires MFCC features for all data..."
  utils/copy_data_dir.sh $data_dir ${data_dir}_hires
  steps/make_mfcc.sh --nj 30 --mfcc-config conf/mfcc_hires.conf \
    --cmd "$train_cmd" ${data_dir}_hires || exit 1;
  steps/compute_cmvn_stats.sh ${data_dir}_hires || exit 1;
  utils/fix_data_dir.sh ${data_dir}_hires
fi

if [ $stage -le 3 ]; then
  echo "$0: extracting i-vectors for all utterances"
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 80 \
    exp/0007_voxceleb_v1_1a ${data_dir}_hires exp/ivectors
fi
exit 1
if [ $stage -le 4 ]; then
  echo "$0: extracting x-vectors for all utterances"
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 80 \
      exp/0007_voxceleb_v2_1a ${data_dir}_hires exp/xvectors
fi

