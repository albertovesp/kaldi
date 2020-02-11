#!/usr/bin/env bash
#set -e
# this script is based on local/multicondition/run_nnet2_common.sh
# minor corrections were made to dir names for nnet3

stage=1
train_set=train
test_sets="test"
gmm=
nnet3_affix=
nj=

set -e
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}

# check if the required tools are present
#local/multi_condition/check_version.sh || exit 1;

mkdir -p exp/nnet3${nnet3_affix}

for f in data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

# low-resolution features and alignments,
if [ -f data/${train_set}/feats.scp ] && [ $stage -le 1 ]; then
  echo "$0: data/${train_set}/feats.scp already exists.  Refusing to overwrite the features "
  echo " to avoid wasting time.  Please remove the file and continue if you really mean this."
  exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: making MFCC features for low-resolution data (needed for alignments)"
  steps/make_mfcc.sh --nj $nj \
    --cmd "$train_cmd" data/${train_set}
  steps/compute_cmvn_stats.sh data/${train_set}
  echo "$0: fixing input data-dir to remove nonexistent features, in case some "
  echo ".. speed-perturbed segments were too short."
  utils/fix_data_dir.sh data/${train_set}
fi

if [ $stage -le 2 ]; then
  if [ -f $ali_dir/ali.1.gz ]; then
    echo "$0: alignments in $ali_dir appear to already exist.  Please either remove them "
    echo " ... or use a later --stage option."
    exit 1
  fi
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/${train_set} data/lang $gmm_dir $ali_dir
fi


# high-resolution features and i-vector extractor,
if [ $stage -le 3 ] && [ -f data/${train_set}_hires/feats.scp ]; then
  echo "$0: data/${train_set}_hires/feats.scp already exists."
  echo " ... Please either remove it, or rerun this script with stage > 3."
  exit 1
fi

if [ $stage -le 3 ]; then
  echo "$0: creating high-resolution MFCC features"

  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  mfccdir=data/${train_set}_hires/data
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/mfcc/aspire-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in ${train_set} ${test_sets}; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/${train_set}_hires

  for datadir in ${train_set} ${test_sets}; do
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires
    steps/compute_cmvn_stats.sh data/${datadir}_hires
    utils/fix_data_dir.sh data/${datadir}_hires
  done
fi

if [ $stage -le 4 ]; then
	# Get subsets of hires data according to previous sets
	utils/subset_data_dir.sh --speakers data/${train_set}_hires 30000 data/${train_set}_30k_hires
	utils/subset_data_dir.sh --speakers data/${train_set}_hires 100000 data/${train_set}_100k_hires
fi

if [ $stage -le 5 ]; then
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    --max-utts 30000 --subsample 2 \
    data/${train_set}_hires exp/nnet3/pca_transform
fi

if [ $stage -le 6 ]; then
  # To train a diagonal UBM we don't need very much data, so use the smallest
  # subset.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 400000 \
    data/${train_set}_30k_hires 512 exp/nnet3${nnet3_affix}/pca_transform \
    exp/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 7 ]; then
  # iVector extractors can in general be sensitive to the amount of data, but
  # this one has a fairly small dim (defaults to 100) so we don't use all of it,
  # we use just the 100k subset (about one sixteenth of the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/${train_set}_100k_hires exp/nnet3/diag_ubm \
    exp/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 8 ]; then
  ivectordir=exp/nnet3/ivectors_${train_set}_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then # this shows how you can split across multiple file-systems.
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/ivectors/aspire/s5/$ivectordir/storage $ivectordir/storage
  fi

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 \
    data/${train_set}_hires data/${train_set}_hires_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 60 \
    data/${train_set}_hires_max2 exp/nnet3${nnet3_affix}/extractor $ivectordir || exit 1;
  
	# Also extract iVectors for the test data, but in this case we don't need the speed
  # perturbation (sp).
  for data in ${test_sets}; do
    nspk=$(wc -l <data/${data}_hires/spk2utt)
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "${nspk}" \
      data/${data}_hires exp/nnet3${nnet3_affix}/extractor \
      exp/nnet3${nnet3_affix}/ivectors_${data}_hires
  done
fi
