#!/bin/bash
# This file copies speaker-level ivectors across frames.

nnet3_affix=
dir=
data=

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

cmd='run.pl'

# get an utterance-level set of iVectors (just duplicate the speaker-level ones).
#utils/apply_map.pl -f 2 $dir/ivectors_onehot.ark <$data/utt2spk >$dir/ivectors_onehot_utt.ark || exit 1;

ivector_dim=$(< $data/spk2utt wc -l)

base_feat_dim=$(feat-to-dim scp:$data/feats.scp -)
start_dim=$base_feat_dim
end_dim=$[$base_feat_dim+$ivector_dim-1]
absdir=$(utils/make_absolute.sh $dir)

ivector_period=10
compress=true

$cmd $dir/log/duplicate_feats.log \
  append-vector-to-feats scp:$data/feats.scp scp:$dir/ivector_online.scp ark:- \| \
  select-feats "$start_dim-$end_dim" ark:- ark:- \| \
  subsample-feats --n=$ivector_period ark:- ark:- \| \
  copy-feats --compress=$compress ark:- \
  ark,scp:$absdir/ivector_online.ark,$absdir/ivector_online.scp || exit 1;
