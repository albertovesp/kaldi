#!/bin/bash

set -e -o pipefail

# This script is called from scripts like local/nnet3/run_tdnn.sh and
# local/chain/run_tdnn.sh (and may eventually be called by more scripts).  It
# contains the common feature preparation and x-vector related parts of the
# script.  See those scripts for examples of usage.


stage=0
nj=30
train_set=train_si284   # you might set this to e.g. train.
test_sets="test_dev93 test_eval92"
gmm=tri4b                # This specifies a GMM-dir from the features of the type you're training the system on;
                         # it should contain alignments for 'train_set'.

#min_seg_len=1.55  # min length in seconds... we do this because chain training
									# will discard segments shorter than 1.5 seconds.  Must remain in sync with
									# the same option given to prepare_lores_feats.sh
xvector_extractor_dir=
lang_dir=
xvector_dim=

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp

for f in data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done


if [ $stage -le 1 ] && [ -f data/${train_set}_sp_xvec/feats.scp ]; then
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
  mfccdir=data/${train_set}_sp_xvec/data
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/mfcc/wsj-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in ${train_set}_sp ${test_sets}; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_xvec
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  #utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_xvec
  for datadir in ${train_set}_sp ${test_sets}; do
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_xvec.conf \
      --cmd "$train_cmd" data/${datadir}_xvec
    steps/compute_cmvn_stats.sh data/${datadir}_xvec
    utils/fix_data_dir.sh data/${datadir}_xvec
  done
fi

if [ $stage -le 2 ]; then
  echo "$0: selecting segments of hires training data that were also present in the"
  echo " ... original training data."

  # note, these data-dirs are temporary; we put them in a sub-directory
  # of the place where we'll make the alignments.
  temp_data_root=exp/nnet3${nnet3_affix}/xvec_${xvector_dim}
  mkdir -p $temp_data_root

  utils/data/subset_data_dir.sh --utt-list data/${train_set}/feats.scp \
          data/${train_set}_sp_xvec $temp_data_root/${train_set}_xvec

  # note: essentially all the original segments should be in the hires data.
  n1=$(wc -l <data/${train_set}/feats.scp)
  n2=$(wc -l <$temp_data_root/${train_set}_xvec/feats.scp)
  if [ $n1 != $n1 ]; then
    echo "$0: warning: number of feats $n1 != $n2, if these are very different it could be bad."
  fi
fi

if [ $stage -le 3 ]; then
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (x-vector starts at zero).
  xvectordir=exp/nnet3${nnet3_affix}/xvectors_${train_set}_sp_hires_${xvector_dim}
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $xvectordir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/xvectors/wsj-$(date +'%m_%d_%H_%M')/s5/$xvectordir/storage $xvectordir/storage
  fi
  temp_data_root=exp/$mic/nnet3${nnet3_affix}/xvec_${xvector_dim}
  mkdir -p $temp_data_root
  
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${train_set}_sp_xvec ${temp_data_root}/${train_set}_sp_xvec_max2

  steps/online/nnet2/extract_xvectors.sh --cmd "$train_cmd" --nj 20 --per-spk false \
    $xvector_extractor_dir ${temp_data_root}/${train_set}_sp_xvec_max2 \
    $xvectordir

  # Also extract x-vectors for the test data, but in this case we don't need the speed
  # perturbation (sp).
  for datadir in ${test_sets}; do
    steps/online/nnet2/extract_xvectors.sh --cmd "$train_cmd" --nj 8 --per-spk false \
      $xvector_extractor_dir data/${datadir}_xvec \
      exp/nnet3${nnet3_affix}/xvectors_${datadir}_hires_${xvector_dim}
  done
fi

exit 0;
