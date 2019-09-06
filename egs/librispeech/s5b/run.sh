#!/bin/bash
# Copyright   2019  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0.

# Trains a DNN for room embeddings generated using RIR transformation.
# We use the 100 hour subset of Librispeech which contains 28539
# utterances. There are 600 room configurations, each containing
# 100 possible speaker and receiver positions. For each utterance,
# we select 8 RIRs randomly, which makes a total
# of 800 hours of training data. This can be changed using the 
# `num_train_replicas` parameter.   

# We keep 50% of the RIRs, i.e, 30000 RIRs in total, which means for the
# case of training with RIR IDs as labels, there would be 30000 labels.
# This can be changed by setting the `keep_frac` parameter to any value 
# between 0 and 1.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc


data=/export/a15/vpanayotov/data
nnet_dir=exp/rvector_nnet_1b

# base url for downloads.
data_url=www.openslr.org/resources/12

# RIR specific configs
rir_dir=data/rir
keep_frac=0.5
num_train_replicas=4

stage=0
train_stage=-1

. utils/parse_options.sh

subsets="dev-clean test-clean train-clean-100"

if [ $stage -le 0 ]; then
  # Download the data. Note that we only download the 100 hour subset 
  # of the training data. If you want to download all the data, you 
  # may also download dev-other, test-other, train-clean-360, and
  # train-other-500. The "clean" subsets are clean and "other" subsets 
  # are noisy. Here we will add RIR noise so we only use the clean
  # subsets.
  for part in $subsets; do
    local/download_and_untar.sh $data $data_url $part
  done
fi

if [ $stage -le 1 ]; then
  # format the data as Kaldi data directories
  for part in dev-clean test-clean train-clean-100; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done
fi

if [ $stage -le 2 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in dev_clean test_clean train_clean_100; do
    steps/make_mfcc.sh --write-utt2dur false --write-utt2num-frames true \
      --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done

	## Uncomment this if you are using larger training data
  # utils/combine_data.sh data/train_clean_460 \
  #  data/train_clean_100 data/train_clean_360
fi

if [ $stage -le 5 ]; then
  # In this stage we create multiple copies of the data using the
  # RIR noise and create corresponding labels. Each generated 
  # utterance has regression labels which correspond to the RIR
  # parameters which were used to generate that augmented utterance
  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  local/prepare_rir_data.sh --stage $stage --rir-dir $rir_dir \
    --keep-frac $keep_frac --num-train-replicas $num_train_replicas 
fi

if [ $stage -le 6 ]; then
  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  for data in "train_rir_100" "dev_rir" "test_rir"; do
    steps/make_mfcc.sh --write-utt2dur false --mfcc-config conf/mfcc.conf \
      --nj 40 --cmd "$train_cmd" \
      data/$data exp/make_mfcc $mfccdir
  done
fi

if [ $stage -le 7 ]; then
  # We need to copy over the vad.scp with some changes since now we
  # have multiple copies of the same utterance with different RIR
  # augmentations.
  old_uttid=data/train_rir_100/old_uttid
  new_uttid=data/train_rir_100/new_uttid
  cut -d " " -f1 data/train_rir_100/wav.scp > $new_uttid
  cut -d "-" -f4- $new_uttid > $old_uttid
  paste $old_uttid $new_uttid | sort -k1 | join - data/train_clean_100/vad.scp -a1 |\
    cut -d " " -f 2,3 | sort -k1 > data/train_rir_100/vad.scp
  rm $old_uttid
  rm $new_uttid 
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 8 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    data/train_rir_100 data/train_rir_100_no_sil exp/train_rir_100_no_sil
  utils/fix_data_dir.sh data/train_rir_100_no_sil 
fi

if [ $stage -le 9 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 4s (400 frames) per utterance.
  min_len=800
  mv data/train_rir_100_no_sil/utt2num_frames data/train_rir_100_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/train_rir_100_no_sil/utt2num_frames.bak > data/train_rir_100_no_sil/utt2num_frames
  utils/filter_scp.pl data/train_rir_100_no_sil/utt2num_frames data/train_rir_100_no_sil/utt2spk > data/train_rir_100_no_sil/utt2spk.new
  mv data/train_rir_100_no_sil/utt2spk.new data/train_rir_100_no_sil/utt2spk
  utils/fix_data_dir.sh data/train_rir_100_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 5 utterances.
  min_num_utts=5
  awk '{print $1, NF-1}' data/train_rir_100_no_sil/spk2utt > data/train_rir_100_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/train_rir_100_no_sil/spk2num | utils/filter_scp.pl - data/train_rir_100_no_sil/spk2utt > data/train_rir_100_no_sil/spk2utt.new
  mv data/train_rir_100_no_sil/spk2utt.new data/train_rir_100_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/train_rir_100_no_sil/spk2utt > data/train_rir_100_no_sil/utt2spk

  utils/filter_scp.pl data/train_rir_100_no_sil/utt2spk data/train_rir_100_no_sil/utt2num_frames > data/train_rir_100_no_sil/utt2num_frames.new
  mv data/train_rir_100_no_sil/utt2num_frames.new data/train_rir_100_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/train_rir_100_no_sil
fi

if [ $stage -le 12 ]; then
  local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage $train_stage \
    --data data/train_rir_100_no_sil --nnet-dir $nnet_dir \
    --egs-dir $nnet_dir/egs
fi

