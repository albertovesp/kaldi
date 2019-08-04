#!/bin/bash

set -e -o pipefail

# This script combines ivectors and xvectors into
# a single speaker embedding.

ivector_dir=
xvector_dir=
out_dir=
nj=

. ./cmd.sh
. utils/parse_options.sh

stage=0

if [ $stage -le 0 ]; then
  # Create single ark file for both
  for dir in $ivector_dir $xvector_dir; do
    for j in $(seq $nj); do cat $dir/ivector_online.$j.ark; done >$dir/ivector_online.ark || exit 1;
  done
fi

if [ $stage -le 1 ]; then
  # Convert vector to matrix

fi

