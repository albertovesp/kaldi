#!/bin/bash

# Copyright     2013  Daniel Povey
#               2019  Desh Raj (Johns Hopkins University)
# Apache 2.0.


# This script computes x-vectors in the same format as extract_ivectors.sh,


# Begin configuration section.
nj=30
cmd="run.pl"
stage=0
num_gselect=5 # Gaussian-selection using diagonal model: number of Gaussians to select
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
ivector_period=10
posterior_scale=0.1 # Scale on the acoustic posteriors, intended to account for
                    # inter-frame correlations.  Making this small during iVector
                    # extraction is equivalent to scaling up the prior, and will
                    # will tend to produce smaller iVectors where data-counts are
                    # small.  It's not so important that this match the value
                    # used when training the iVector extractor, but more important
                    # that this match the value used when you do real online decoding
                    # with the neural nets trained with these iVectors.
max_count=100       # Interpret this as a number of frames times posterior scale...
                    # this config ensures that once the count exceeds this (i.e.
                    # 1000 frames, or 10 seconds, by default), we start to scale
                    # down the stats, accentuating the prior term.   This seems quite
                    # important for some reason.
sub_speaker_frames=0  # If >0, during iVector estimation we split each speaker
                      # into possibly many 'sub-speakers', each with at least
                      # this many frames of speech (evaluated after applying
                      # silence_weight, so will typically exclude silence.
                      # e.g. set this to 1000, and it will require at least 10 seconds
                      # of speech per sub-speaker.

compress=true       # If true, compress the iVectors stored on disk (it's lossy
                    # compression, as used for feature matrices).
silence_weight=0.0
acwt=0.1  # used if input is a decode dir, to get best path from lattices.
mdl=final  # change this if decode directory did not have ../final.mdl present.
num_threads=1 # Number of threads used by ivector-extract.  It is usually not
              # helpful to set this to > 1.  It is only useful if you have
              # fewer speakers than the number of jobs you want to run.
cache_capacity=64 # Cache capacity for x-vector extractor
chunk_size=-1     # The chunk size over which the embedding is extracted.
                  # If left unspecified, it uses the max_chunk_size in the nnet
                  # directory.
use_gpu=false

# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ] && [ $# != 5 ]; then
  echo "Usage: $0 [options] <data> <lang> <extractor-dir> [<alignment-dir>|<decode-dir>|<weights-archive>] <ivector-dir>"
  echo " e.g.: $0 data/test data/lang exp/nnet2_online/extractor exp/tri3/decode_test exp/nnet2_online/ivectors_test"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-threads)"
  echo "  --num-threads <n|1>                              # Number of threads for each job"
  echo "                                                   # Ignored if <alignment-dir> or <decode-dir> supplied."
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --num-gselect <n|5>                              # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  echo "  --min-post <float;default=0.025>                 # Pruning threshold for posteriors"
  echo "  --ivector-period <int;default=10>                # How often to extract an iVector (frames)"
  echo "  --posterior-scale <float;default=0.1>            # Scale on posteriors in iVector extraction; "
  echo "                                                   # affects strength of prior term."

  exit 1;
fi

if [ $# -eq 4 ]; then
  data=$1
  lang=$2
  srcdir=$3
  dir=$4
else # 5 arguments
  data=$1
  lang=$2
  srcdir=$3
  ali_or_decode_dir=$4
  dir=$5
fi

for f in $srcdir/final.raw $srcdir/min_chunk_size $srcdir/max_chunk_size $data/feats.scp; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

mkdir -p $dir/log
silphonelist=$(cat $lang/phones/silence.csl) || exit 1;

if [ ! -z "$ali_or_decode_dir" ]; then


  if [ -f $ali_or_decode_dir/ali.1.gz ]; then
    if [ ! -f $ali_or_decode_dir/${mdl}.mdl ]; then
      echo "$0: expected $ali_or_decode_dir/${mdl}.mdl to exist."
      exit 1;
    fi
    nj_orig=$(cat $ali_or_decode_dir/num_jobs) || exit 1;

    if [ $stage -le 0 ]; then
      rm $dir/weights.*.gz 2>/dev/null

      $cmd JOB=1:$nj_orig  $dir/log/ali_to_post.JOB.log \
        gunzip -c $ali_or_decode_dir/ali.JOB.gz \| \
        ali-to-post ark:- ark:- \| \
        weight-silence-post $silence_weight $silphonelist $ali_or_decode_dir/final.mdl ark:- ark:- \| \
        post-to-weights ark:- "ark:|gzip -c >$dir/weights.JOB.gz" || exit 1;

      # put all the weights in one archive.
      for j in $(seq $nj_orig); do gunzip -c $dir/weights.$j.gz; done | gzip -c >$dir/weights.gz || exit 1;
      rm $dir/weights.*.gz || exit 1;
    fi

  elif [ -f $ali_or_decode_dir/lat.1.gz ]; then
    nj_orig=$(cat $ali_or_decode_dir/num_jobs) || exit 1;
    if [ ! -f $ali_or_decode_dir/../${mdl}.mdl ]; then
      echo "$0: expected $ali_or_decode_dir/../${mdl}.mdl to exist."
      exit 1;
    fi


    if [ $stage -le 0 ]; then
      rm $dir/weights.*.gz 2>/dev/null

      $cmd JOB=1:$nj_orig  $dir/log/lat_to_post.JOB.log \
        lattice-best-path --acoustic-scale=$acwt "ark:gunzip -c $ali_or_decode_dir/lat.JOB.gz|" ark:/dev/null ark:- \| \
        ali-to-post ark:- ark:- \| \
        weight-silence-post $silence_weight $silphonelist $ali_or_decode_dir/../${mdl}.mdl ark:- ark:- \| \
        post-to-weights ark:- "ark:|gzip -c >$dir/weights.JOB.gz" || exit 1;

      # put all the weights in one archive.
      for j in $(seq $nj_orig); do gunzip -c $dir/weights.$j.gz; done | gzip -c >$dir/weights.gz || exit 1;
      rm $dir/weights.*.gz || exit 1;
    fi
  elif [ -f $ali_or_decode_dir ] && gunzip -c $ali_or_decode_dir >/dev/null; then
    cp $ali_or_decode_dir $dir/weights.gz || exit 1;
  else
    echo "$0: expected ali.1.gz or lat.1.gz to exist in $ali_or_decode_dir";
    exit 1;
  fi
fi

sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

echo $ivector_period > $dir/ivector_period || exit 1;
splice_opts=$(cat $srcdir/splice_opts)

# Set up the features
feats="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:${sdata}/feats.scp ark:- | "

if [ $sub_speaker_frames -gt 0 ]; then

  if [ $stage -le 1 ]; then
  # We work out 'fake' spk2utt files that possibly split each speaker into multiple pieces.
    if [ ! -z "$ali_or_decode_dir" ]; then
      gunzip -c $dir/weights.gz | copy-vector ark:- ark,t:- | \
        awk '{ sum=0; for (n=3;n<NF;n++) sum += $n; print $1, sum; }' > $dir/utt_counts || exit 1;
    else
      feat-to-len scp:$data/feats.scp ark,t:- > $dir/utt_counts || exit 1;
    fi
    if ! [ $(wc -l <$dir/utt_counts) -eq $(wc -l <$data/feats.scp) ]; then
      echo "$0: error getting per-utterance counts."
      exit 0;
    fi
    cat $data/spk2utt | python -c "
import sys
utt_counts = {}
trash = list(map(lambda x: utt_counts.update({x.split()[0]:float(x.split()[1])}), open('$dir/utt_counts').readlines()))
sub_speaker_frames = $sub_speaker_frames
lines = sys.stdin.readlines()
total_counts = {}
for line in lines:
  parts = line.split()
  spk = parts[0]
  total_counts[spk] = 0
  for utt in parts[1:]:
    total_counts[spk] += utt_counts[utt]

for line_index in range(len(lines)):
  line = lines[line_index]
  parts = line.split()
  spk = parts[0]

  numeric_id=0
  current_count = 0
  covered_count = 0
  current_utts = []
  for utt in parts[1:]:
    try:
      current_count += utt_counts[utt]
      covered_count += utt_counts[utt]
    except KeyError:
      raise Exception('No count found for the utterance {0}.'.format(utt))
    current_utts.append(utt)
    if ((current_count >= $sub_speaker_frames) and ((total_counts[spk] - covered_count) >= $sub_speaker_frames)) or (utt == parts[-1]):
      spk_partial = '{0}-{1:06x}'.format(spk, numeric_id)
      numeric_id += 1
      print ('{0} {1}'.format(spk_partial, ' '.join(current_utts)))
      current_utts = []
      current_count = 0
"> $dir/spk2utt || exit 1;
    mkdir -p $dir/split$nj
    # create split versions of our spk2utt file.
    for j in $(seq $nj); do
      mkdir -p $dir/split$nj/$j
      utils/filter_scp.pl -f 2 $sdata/$j/utt2spk <$dir/spk2utt >$dir/split$nj/$j/spk2utt || exit 1;
      utils/spk2utt_to_utt2spk.pl <$dir/split$nj/$j/spk2utt >$dir/split$nj/$j/utt2spk || exit 1;
    done
  fi
  this_sdata=$dir/split$nj
else
  this_sdata=$sdata
fi

if [ $stage -le 2 ]; then
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

# get an utterance-level set of iVectors (just duplicate the speaker-level ones).
# note: if $this_sdata is set $dir/split$nj, then these won't be real speakers, they'll
# be "sub-speakers" (speakers split up into multiple utterances).
if [ $stage -le 3 ]; then
  for j in $(seq $nj); do
    utils/apply_map.pl -f 2 $dir/ivectors_spk.$j.ark <$this_sdata/$j/utt2spk >$dir/ivectors_utt.$j.ark || exit 1;
  done
fi

xvector_dim=$[$(head -n 1 $dir/ivectors_spk.1.ark | wc -w) - 3] || exit 1;
echo  "$0: x-vector dim is $xvector_dim"

base_feat_dim=$(feat-to-dim scp:$data/feats.scp -) || exit 1;

start_dim=$base_feat_dim
end_dim=$[$base_feat_dim+$xvector_dim-1]
absdir=$(utils/make_absolute.sh $dir)

if [ $stage -le 4 ]; then
  # here, we are just using the original features in $sdata/JOB/feats.scp for
  # their number of rows; we use the select-feats command to remove those
  # features and retain only the iVector features.
  $cmd JOB=1:$nj $dir/log/duplicate_feats.JOB.log \
    append-vector-to-feats scp:$sdata/JOB/feats.scp ark:$dir/ivectors_utt.JOB.ark ark:- \| \
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

