#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

# This script trains a DNN similar to the recipe described in
# http://www.danielpovey.com/files/2017_interspeech_embeddings.pdf .

. ./cmd.sh
set -e

stage=1
train_stage=0
use_gpu=true
remove_egs=false

data=data/train
nnet_dir=exp/xvector_nnet_1a/
egs_dir=exp/xvector_nnet_1a/egs

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

num_pdfs=$(awk '{print $2}' $data/utt2spk | sort | uniq -c | wc -l)

# Now we create the nnet examples using sid/nnet3/xvector/get_egs.sh.
# The argument --num-repeats is related to the number of times a speaker
# repeats per archive.  If it seems like you're getting too many archives
# (e.g., more than 200) try increasing the --frames-per-iter option.  The
# arguments --min-frames-per-chunk and --max-frames-per-chunk specify the
# minimum and maximum length (in terms of number of frames) of the features
# in the examples.
#
# To make sense of the egs script, it may be necessary to put an "exit 1"
# command immediately after stage 3.  Then, inspect
# exp/<your-dir>/egs/temp/ranges.* . The ranges files specify the examples that
# will be created, and which archives they will be stored in.  Each line of
# ranges.* has the following form:
#    <utt-id> <local-ark-indx> <global-ark-indx> <start-frame> <end-frame> <spk-id>
# For example:
#    100304-f-sre2006-kacg-A 1 2 4079 881 23

# If you're satisfied with the number of archives (e.g., 50-150 archives is
# reasonable) and with the number of examples per speaker (e.g., 1000-5000
# is reasonable) then you can let the script continue to the later stages.
# Otherwise, try increasing or decreasing the --num-repeats option.  You might
# need to fiddle with --frames-per-iter.  Increasing this value decreases the
# the number of archives and increases the number of examples per archive.
# Decreasing this value increases the number of archives, while decreasing the
# number of examples per archive.
if [ $stage -le 4 ]; then
  echo "$0: Getting neural network training egs";
  # dump egs.
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $egs_dir/storage ]; then
    utils/create_split_dir.pl \
     /export/b{03,04,05,06}/$USER/kaldi-data/egs/sre16/v2/xvector-$(date +'%m_%d_%H_%M')/$egs_dir/storage $egs_dir/storage
  fi
  sid/nnet3/xvector/get_egs.sh --cmd "$cuda_cmd" \
    --nj 8 \
    --stage 0 \
    --num-heldout-utts 10000 \
    --frames-per-iter 1000000000 \
    --frames-per-iter-diagnostic 10000000 \
    --min-frames-per-chunk 200 \
    --max-frames-per-chunk 400 \
    --num-diagnostic-archives 4 \
    --num-repeats 35 \
    "$data" $egs_dir
fi

if [ $stage -le 5 ]; then
  num_train_archives=142
  num_valid_archives=4
  mkdir -p exp/xvector_pytorch/egs/tmp
  mkdir -p exp/xvector_pytorch/egs/log

  scripts/create_segments_file.py exp/xvector_nnet_1a/egs exp/xvector_pytorch/egs/tmp

  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d exp/xvector_pytorch/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b{03,04,05,06}/$USER/kaldi-data/egs/sre16/v2/xvector-$(date +'%m_%d_%H_%M')/xvector_pytorch/storage exp/xvector_pytorch/egs/storage
    utils/create_data_link.pl $(for x in $(seq $num_train_archives); do echo exp/xvector_pytorch/egs/egs.$x.ark; done)
  fi

  # Create the training examples
  queue.pl --mem 4G --max-jobs-run 30 JOB=1:$num_train_archives exp/xvector_pytorch/egs/log/extract_segment.JOB.log \
      extract-rows exp/xvector_pytorch/egs/tmp/egs.JOB.scp scp:data/swbd_sre_combined_no_sil/feats.scp ark,scp:exp/xvector_pytorch/egs/egs.JOB.ark,exp/xvector_pytorch/egs/egs.JOB.scp || exit 1;

  # Create the validation examples
  queue.pl --mem 4G JOB=1:$num_valid_archives exp/xvector_pytorch/egs/log/extract_segment_valid.JOB.log \
      extract-rows exp/xvector_pytorch/egs/tmp/valid_egs.JOB.scp scp:data/swbd_sre_combined_no_sil/feats.scp ark,scp:exp/xvector_pytorch/egs/valid_egs.JOB.ark,exp/xvector_pytorch/egs/valid_egs.JOB.scp || exit 1;
fi

if [ $stage -le 6 ]; then
  source ~/.bashrc
  exp_dir="exp/test_exp/"
  mkdir -p $exp_dir/log
  mkdir -p $exp_dir/model
  queue.pl --mem 20G --gpu 1 $exp_dir/log/train.log scripts/train.py \
    --epochs 3 \
    --start-epoch 0 \
    --print-freq 200 \
    --batch-size 128 \
    --lr 0.01 \
    $exp_dir
fi

exit 0;
