#!/usr/bin/env bash
# Copyright   2019   Ashish Arora, Vimal Manohar
# Apache 2.0.
# This script performs decoding on a data directory generated after diarization
# and GSS-based enhancement.

stage=2
nj=4
cmd=run.pl
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
if [ $# != 4 ]; then
  echo "Usage: $0 <in-data-dir> <lang-dir> <model-dir> <ivector-dir>"
  echo "e.g.: $0 data/dev data/lang_chain exp/chain_train_worn_simu_u400k_cleaned_rvb \
                 exp/nnet3_train_worn_simu_u400k_cleaned_rvb"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs."
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

data_dir=$1
lang_dir=$2
asr_model_dir=$3
ivector_extractor=$4

for f in $data_dir/wav.scp \
         $lang_dir/L.fst $asr_model_dir/tree_sp/graph/HCLG.fst \
         $asr_model_dir/tdnn1b_cnn_sp/final.mdl; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

if [ $stage -le 1 ]; then
  echo "$0 extracting MFCCs for data"
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $nj --cmd $cmd ${data_dir}
  steps/compute_cmvn_stats.sh ${data_dir}
  utils/fix_data_dir.sh ${data_dir}
fi

if [ $stage -le 2 ]; then
  echo "$0 performing decoding on the extracted features"
  local/nnet3/decode.sh --affix 2stage --acwt 1.0 --post-decode-acwt 10.0 \
    --frames-per-chunk 150 --nj $nj --ivector-dir $ivector_extractor \
    $data_dir $lang_dir $asr_model_dir/tree_sp/graph $asr_model_dir/tdnn1b_cnn_sp/
fi

if [ $stage -le 3 ]; then
  data_set=$(basename $data_dir)
  while read -r line;
  do
    utteranceid=$(echo "$line" | cut -f1 -d " ")
    speakerid=$(echo "$line" | cut -f1 -d "_")
    text=$(echo "$line" | cut -f2- -d " ")
    echo $utteranceid "  " $speakerid "  " $text
  done < $asr_model_dir/tdnn1b_cnn_sp/decode_${data_set}_2stage/scoring_kaldi/penalty_0.0/10.txt
fi
exit 0
