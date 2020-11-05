#!/bin/bash

# Copyright       2016  David Snyder
#            2017-2018  Matthew Maciejewski
#                 2020  Maxim Korenevsky (STC-innovations Ltd)
# Apache 2.0.

# This script performs spectral clustering using scored
# pairs of subsegments and produces a rttm file with speaker
# labels derived from the clusters.

# Begin configuration section.
cmd="run.pl"
stage=0
nj=10
cleanup=true
rttm_channel=0
reco2num_spk=
overlap_rttm=  # Path to an RTTM output of an external overlap detector
rttm_affix=

# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0 <src-dir> <dir>"
  echo " e.g.: $0 exp/ivectors_callhome exp/ivectors_callhome/results"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --rttm-channel <rttm-channel|0>                  # The value passed into the RTTM channel field. Only affects"
  echo "                                                   # the format of the RTTM file."
  echo "  --reco2num-spk <reco2num-spk-file>               # File containing mapping of recording ID"
  echo "                                                   # to number of speakers. Used instead of threshold"
  echo "                                                   # as stopping criterion if supplied."
  echo "  --overlap-rttm <overlap-rttm-file>               # File containing overlap segments"
  echo "  --cleanup <bool|false>                           # If true, remove temporary files"
  exit 1;
fi

srcdir=$1
dir=$2

reco2num_spk_opts=
if [ ! $reco2num_spk == "" ]; then
  reco2num_spk_opts="--reco2num-spk $reco2num_spk"
fi

mkdir -p $dir/tmp

for f in $srcdir/scores.scp $srcdir/spk2utt $srcdir/utt2spk $srcdir/segments ; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# We use a different Python version in which the local
# scikit-learn is installed.
miniconda_dir=$HOME/miniconda3/
if [ ! -d $miniconda_dir ]; then
    echo "$miniconda_dir does not exist. Please run '$KALDI_ROOT/tools/extras/install_miniconda.sh'."
    exit 1
fi

overlap_rttm_opt=
if ! [ -z "$overlap_rttm" ]; then
  overlap_rttm_opt="--overlap_rttm $overlap_rttm"
  sc_bin="spec_clust_overlap.py"
  rttm_bin="make_rttm_ol.py"
  # Install a modified version of scikit-learn using:
  echo "The overlap-aware spectral clustering requires installing a modified version\n"
  echo "of scitkit-learn. You can download it using:\n"
  echo "$miniconda_dir/bin/python -m pip install git+https://github.com/desh2608/scikit-learn.git@overlap \n"
  echo "if the process fails while clustering."
else
  sc_bin="spec_clust.py"
  rttm_bin="make_rttm.py"
fi

cp $srcdir/spk2utt $dir/tmp/
cp $srcdir/utt2spk $dir/tmp/
cp $srcdir/segments $dir/tmp/
utils/fix_data_dir.sh $dir/tmp > /dev/null

if [ ! -z $reco2num_spk ]; then
  reco2num_spk="ark,t:$reco2num_spk"
fi

sdata=$dir/tmp/split$nj;
utils/split_data.sh $dir/tmp $nj || exit 1;

# Set various variables.
mkdir -p $dir/log

feats="utils/filter_scp.pl $sdata/JOB/spk2utt $srcdir/scores.scp |"

reco2num_spk_opt=
if [ ! $reco2num_spk == "" ]; then
  reco2num_spk_opt="--reco2num_spk $reco2num_spk"
fi

if [ $stage -le 0 ]; then
  echo "$0: clustering scores"
  for j in `seq $nj`; do 
    utils/filter_scp.pl $sdata/$j/spk2utt $srcdir/scores.scp > $dir/scores.$j.scp
  done
  $cmd JOB=1:$nj $dir/log/spectral_cluster.JOB.log \
    $miniconda_dir/bin/python diarization/$sc_bin $reco2num_spk_opt $overlap_rttm_opt \
      scp:$dir/scores.JOB.scp ark,t:$sdata/JOB/spk2utt ark,t:$dir/labels.JOB || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: combining labels"
  for j in $(seq $nj); do cat $dir/labels.$j; done > $dir/labels || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0: computing RTTM"
  diarization/$rttm_bin --rttm-channel $rttm_channel $srcdir/segments $dir/labels $dir/rttm${rttm_affix} || exit 1;
fi

if $cleanup ; then
  rm -r $dir/tmp || exit 1;
fi
