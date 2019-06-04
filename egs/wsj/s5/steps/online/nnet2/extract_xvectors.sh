#!/bin/bash

# Copyright     2017  David Snyder
#               2017  Johns Hopkins University (Author: Daniel Povey)
#               2017  Johns Hopkins University (Author: Daniel Garcia Romero)
# Apache 2.0.

# This script extracts embeddings (called "xvectors" here) from a set of
# utterances, given features and a trained DNN.  The purpose of this script
# is analogous to sid/extract_ivectors.sh: it creates archives of
# vectors that are used in speaker recognition.  Like ivectors, xvectors can
# be used in PLDA or a similar backend for scoring.

# Begin configuration section.
nj=30
cmd="run.pl"

cache_capacity=64 # Cache capacity for x-vector extractor
chunk_size=-1     # The chunk size over which the embedding is extracted.
                  # If left unspecified, it uses the max_chunk_size in the nnet
                  # directory.
compress=true       # If true, compress the iVectors stored on disk (it's lossy
                    # compression, as used for feature matrices).
ivector_period=10
sub_speaker_frames=0  # If >0, during iVector estimation we split each speaker
                      # into possibly many 'sub-speakers', each with at least
                      # this many frames of speech (evaluated after applying
                      # silence_weight, so will typically exclude silence.
                      # e.g. set this to 1000, and it will require at least 10 seconds
                      # of speech per sub-speaker.
use_gpu=false
stage=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <nnet-dir> <data> <xvector-dir>"
  echo " e.g.: $0 exp/xvector_nnet data/train exp/xvectors_train"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --cache-capacity <n|64>                          # To speed-up xvector extraction"
  echo "  --chunk-size <n|-1>                              # If provided, extracts embeddings with specified"
  echo "                                                   # chunk size, and averages to produce final embedding"
fi

srcdir=$1
data=$2
dir=$3

for f in $srcdir/final.raw $srcdir/min_chunk_size $srcdir/max_chunk_size $data/feats.scp ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

min_chunk_size=`cat $srcdir/min_chunk_size 2>/dev/null`
max_chunk_size=`cat $srcdir/max_chunk_size 2>/dev/null`

nnet=$srcdir/final.raw
if [ -f $srcdir/extract.config ] ; then
  echo "$0: using $srcdir/extract.config to extract xvectors"
  nnet="nnet3-copy --nnet-config=$srcdir/extract.config $srcdir/final.raw - |"
fi

if [ $chunk_size -le 0 ]; then
  chunk_size=$max_chunk_size
fi

if [ $max_chunk_size -lt $chunk_size ]; then
  echo "$0: specified chunk size of $chunk_size is larger than the maximum chunk size, $max_chunk_size" && exit 1;
fi

mkdir -p $dir/log

utils/split_data.sh $data $nj
echo "$0: extracting xvectors for $data"
sdata=$data/split$nj/JOB

# Set up the features
feat="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:${sdata}/feats.scp ark:- |"

if [ $stage -le 0 ]; then
  echo "$0: extracting xvectors from nnet"
  if $use_gpu; then
    for g in $(seq $nj); do
      $cmd --gpu 1 ${dir}/log/extract.$g.log \
        nnet3-xvector-compute --use-gpu=yes --min-chunk-size=$min_chunk_size --chunk-size=$chunk_size --cache-capacity=${cache_capacity} \
        "$nnet" "`echo $feat | sed s/JOB/$g/g`" ark,scp:${dir}/xvector.$g.ark,${dir}/xvector.$g.scp || exit 1 &
    done
    wait
  else
    $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
      nnet3-xvector-compute --use-gpu=no --min-chunk-size=$min_chunk_size --chunk-size=$chunk_size --cache-capacity=${cache_capacity} \
      "$nnet" "$feat" ark,scp:${dir}/xvector.JOB.ark,${dir}/xvector.JOB.scp || exit 1;
  fi
fi

if [ $stage -le 1 ]; then
  echo "$0: combining xvectors across jobs"
  # Note that we name it ivector_online.scp to avoid changing other scripts
  for j in $(seq $nj); do cat $dir/xvector.$j.scp; done >$dir/ivector_online.scp || exit 1;
fi

if [ $stage -le 2 ]; then
  # Average the utterance-level xvectors to get speaker-level xvectors.
  echo "$0: computing mean of xvectors for each speaker"
  $cmd JOB=1:$nj $dir/log/speaker_mean.JOB.log \
    ivector-mean ark:$sdata/spk2utt scp:$dir/xvector.JOB.scp \
    ark,t:$dir/ivectors_spk.JOB.ark ark,t:$dir/num_utts.JOB.ark || exit 1;
fi

this_sdata=$data/split$nj/
# get an utterance-level set of iVectors (just duplicate the speaker-level ones).
# note: if $this_sdata is set $dir/split$nj, then these won't be real speakers, they'll
# be "sub-speakers" (speakers split up into multiple utterances).
if [ $stage -le 3 ]; then
  for j in $(seq $nj); do
    utils/apply_map.pl -f 2 $dir/ivectors_spk.$j.ark <$this_sdata/$j/utt2spk >$dir/ivectors_utt.$j.ark || exit 1;
  done
fi

ivector_dim=$[$(head -n 1 $dir/ivectors_spk.1.ark | wc -w) - 3] || exit 1;
echo  "$0: iVector dim is $ivector_dim"

base_feat_dim=$(feat-to-dim scp:$data/feats.scp -) || exit 1;

start_dim=$base_feat_dim
end_dim=$[$base_feat_dim+$ivector_dim-1]
absdir=$(utils/make_absolute.sh $dir)

if [ $stage -le 4 ]; then
  # here, we are just using the original features in $sdata/JOB/feats.scp for
  # their number of rows; we use the select-feats command to remove those
  # features and retain only the iVector features.
  $cmd JOB=1:$nj $dir/log/duplicate_feats.JOB.log \
    append-vector-to-feats scp:$sdata/feats.scp ark:$dir/ivectors_utt.JOB.ark ark:- \| \
    select-feats "$start_dim-$end_dim" ark:- ark:- \| \
    subsample-feats --n=$ivector_period ark:- ark:- \| \
    copy-feats --compress=$compress ark:- \
    ark,scp:$absdir/ivector_online.JOB.ark,$absdir/ivector_online.JOB.scp || exit 1;
fi

if [ $stage -le 5 ]; then
  echo "$0: combining iVectors across jobs"
  for j in $(seq $nj); do cat $dir/ivector_online.$j.scp; done >$dir/ivector_online.scp || exit 1;
fi

steps/nnet2/get_ivector_id.sh $srcdir > $dir/final.ie.id || exit 1

echo "$0: done extracting (pseudo-online) iVectors to $dir using the extractor in $srcdir."

