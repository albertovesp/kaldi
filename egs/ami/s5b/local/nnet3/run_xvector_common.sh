#!/bin/bash

set -e -o pipefail

# This script is called from scripts like local/nnet3/run_tdnn.sh and
# local/chain/run_tdnn.sh (and may eventually be called by more scripts).  It
# contains the common feature preparation and x-vector related parts of the
# script.  See those scripts for examples of usage.


stage=0
mic=ihm
nj=30
min_seg_len=1.55  # min length in seconds... we do this because chain training
                  # will discard segments shorter than 1.5 seconds.  Must remain in sync with
                  # the same option given to prepare_lores_feats.sh.
train_set=train_cleaned   # you might set this to e.g. train_cleaned.
gmm=tri3          # This specifies a GMM-dir from the features of the type you're training the system on;
                  # it should contain alignments for 'train_set'.

extractor_dir=
nnet3_affix=_cleaned     # affix for exp/$mic/nnet3 directory to put iVector stuff in, so it
                         # becomes exp/$mic/nnet3_cleaned or whatever.

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

test_sets="dev eval"
gmmdir=exp/${mic}/${gmm}

for f in data/${mic}/${train_set}/feats.scp ; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if [ $stage -le 1 ] && [ -f data/${mic}/${train_set}_sp_xvec/feats.scp ]; then
  echo "$0: data/${train_set}_sp_xvec/feats.scp already exists."
  echo " ... Please either remove it, or rerun this script with stage > 1."
  exit 1
fi

if [ $stage -le 1 ]; then
  echo "$0: creating MFCC features for x-vector extractor"

  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  mfccdir=data/${mic}/${train_set}_sp_xvec/data
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/mfcc/ami-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in ${train_set}_sp ${test_sets}; do
    utils/copy_data_dir.sh data/${mic}/$datadir data/${mic}/${datadir}_xvec
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  #utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_xvec
  for datadir in ${train_set}_sp ${test_sets}; do
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_xvec.conf \
      --cmd "$train_cmd" data/${mic}/${datadir}_xvec
    steps/compute_cmvn_stats.sh data/${mic}/${datadir}_xvec
    utils/fix_data_dir.sh data/${mic}/${datadir}_xvec
  done
fi

if [ $stage -le 2 ]; then
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (x-vector starts at zero).
  xvectordir=exp/${mic}/nnet3${nnet3_affix}/xvectors_${train_set}_sp_hires_sre
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $xvectordir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/xvectors/ami-$(date +'%m_%d_%H_%M')/s5/$xvectordir/storage $xvectordir/storage
  fi

  steps/online/nnet2/extract_xvectors.sh --cmd "$train_cmd" --nj 20 --per-spk false \
    --length-normalize true \
    $extractor_dir data/${mic}/${train_set}_sp_xvec \
    $xvectordir

  # Also extract x-vectors for the test data, but in this case we don't need the speed
  # perturbation (sp).
  for datadir in ${test_sets}; do
    steps/online/nnet2/extract_xvectors.sh --cmd "$train_cmd" --nj 20 --per-spk false \
      --length-normalize true \
      $extractor_dir data/${mic}/${datadir}_xvec \
      exp/${mic}/nnet3${nnet3_affix}/xvectors_${datadir}_hires_sre
  done
fi

exit 0;
