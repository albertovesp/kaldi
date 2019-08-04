#!/bin/bash

set -e -o pipefail

# This script applies PCA transform on the extracted
# x-vectors and converts them to a form that is suitable
# for use with ASR systems. It is called from 
# run_tdnn.sh scripts which use x-vectors for speaker
# adaptation.

dir=
dir_reduced=
pca_dim=100
data=
nj=

. ./cmd.sh
. utils/parse_options.sh

stage=0
ivector_period=10
compress=true

if [ $stage -le 0 ]; then
	# We have to use utt-level x-vectors since the ivector_online.scp
	# file is not in correct format for PCA.
  echo "$0: combining utt xvectors across jobs"
  for j in $(seq $nj); do cat $dir/xvector.$j.ark; done >$dir/xvector.ark || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: Computing whitening transform"
  $train_cmd $dir_reduced/log/transform.log \
    est-pca --read-vectors=true --normalize-mean=false \
      --normalize-variance=true --dim=$pca_dim \
      ark:$dir/xvector.ark $dir_reduced/transform.mat || exit 1;
	
	$train_cmd $dir/log/pca.log \
		ivector-transform $dir_reduced/transform.mat ark:$dir/xvector.ark ark:- \| \
    ivector-normalize-length ark:- ark,scp:$dir_reduced/xvector.ark,$dir_reduced/xvector.scp
fi

# We convert these new x-vectors back to the format
# required to train in the ASR DNN system.
if [ $stage -le 2 ]; then
	base_feat_dim=$(feat-to-dim scp:$data/feats.scp -) || exit 1;

	start_dim=$base_feat_dim
	end_dim=$[$base_feat_dim+$pca_dim-1]
	absdir=$(utils/make_absolute.sh $dir_reduced)

  $train_cmd $dir/log/duplicate_feats.log \
    append-vector-to-feats scp:$data/feats.scp ark:$dir_reduced/xvector.ark ark:- \| \
    select-feats "$start_dim-$end_dim" ark:- ark:- \| \
    subsample-feats --n=$ivector_period ark:- ark:- \| \
    copy-feats --compress=$compress ark:- \
    ark,scp:$absdir/ivector_online.ark,$absdir/ivector_online.scp || exit 1;

fi
