#!/bin/bash

# Copyright 2014  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
# This is similar to prepare_online_decoding.sh, but it uses a new Bayesian
# method for estimating the utterance-level average of speech and noise 
# frames, which is concatenated and used as an alternate to i-vectors.

# Begin configuration.
stage=0 # This allows restarting after partway, when something when wrong.
feature_type=mfcc
add_pitch=false
mfcc_config=conf/mfcc.conf # you can override any of these you need to override.
plp_config=conf/plp.conf
fbank_config=conf/fbank.conf

# online_pitch_config is the config file for both pitch extraction and
# post-processing; we combine them into one because during training this
# is given to the program compute-and-process-kaldi-pitch-feats.
online_pitch_config=conf/online_pitch.conf

ivector_period=10 # Number of frames for which the i-vector stays the same
                  # (use same value as from local/nnet3/run_ivector_common.sh).
iter=final
# End configuration.

cmd=run.pl

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# -ne 5 ]; then
   echo "Usage: $0 [options] <data-dir> <lang-dir> <noise-prior-file> <nnet-dir> <output-dir>"
   echo "e.g.: $0 data/train data/lang exp/noise_prior exp/nnet3 exp/nnet3_online"
   echo "main options (for others, see top of script file)"
   echo "  --feature-type <mfcc|plp>                        # Type of the base features; "
   echo "                                                   # important to generate the correct"
   echo "                                                   # configs in <output-dir>/conf/"
   echo "  --add-pitch <true|false>                         # Append pitch features to cmvn"
   echo "                                                   # (default: false)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --config <config-file>                           # config containing options"
   echo "  --iter <model-iteration|final>                   # iteration of model to take."
   echo "  --stage <stage>                                  # stage to do partial re-run from."
   exit 1;
fi


if [ $# -eq 5 ]; then
  data=$1
  lang=$2
  npfile=$3
  srcdir=$4
  dir=$5
fi

data=${data}_sp_hires

for f in $lang/phones/silence.csl $srcdir/${iter}.mdl $srcdir/tree \
  $data/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
if [ ! -z "$npfile" ]; then
  [ ! -f $npfile ] && echo "$0: no such file $npfile" && exit 1;
fi


dir=$(utils/make_absolute.sh $dir) # Convert $dir to an absolute pathname, so that the
                        # configuration files we write will contain absolute
                        # pathnames.
mkdir -p $dir/conf
mkdir -p $dir/log

cp $lang/phones.txt $dir || exit 1;

silphonelist=`cat $lang/phones/silence.csl` || exit 1;

rm $dir/{plp,mfcc,fbank}.conf 2>/dev/null
echo "$0: preparing configuration files in $dir/conf"

if [ -f $dir/conf/online.conf ]; then
  echo "$0: moving $dir/conf/online.conf to $dir/conf/online.conf.bak"
  mv $dir/conf/online.conf $dir/conf/online.conf.bak
fi

conf=$dir/conf/online.conf
echo -n >$conf

if $add_pitch; then
  echo "$0: enabling pitch features"
  echo "--add-pitch=true" >>$conf
  echo "$0: creating $dir/conf/online_pitch.conf"
  if [ ! -f $online_pitch_config ]; then
    echo "$0: expected file '$online_pitch_config' to exist.";
    exit 1;
  fi
  cp $online_pitch_config $dir/conf/online_pitch.conf || exit 1;
  echo "--online-pitch-config=$dir/conf/online_pitch.conf" >>$conf
fi

if $add_pitch; then
  skip_opt="--skip-dims=13:14:15" # should make this more general.
fi

if [ $stage -le 1 ]; then
  case "$feature_type" in
    mfcc)
      echo "$0: creating $dir/conf/mfcc.conf"
      echo "--mfcc-config=$dir/conf/mfcc.conf" >>$conf
      cp conf/mfcc_hires.conf $dir/conf/mfcc.conf ;;
    *)
      echo "Unknown feature type $feature_type"
  esac
  if $add_pitch; then
    echo "$0: enabling pitch features"
    echo "--add-pitch" >>$conf
    echo "$0: creating $dir/conf/pitch.conf"
    echo "--pitch-config=$dir/conf/pitch.conf" >>$conf
    if ! cp $pitch_config $dir/conf/pitch.conf; then
      echo "$0: error copying pitch config to $dir/conf/"
      exit 1;
    fi;
    echo "$0: creating $dir/conf/pitch_process.conf"
    echo "--pitch-process-config=$dir/conf/pitch_process.conf" >>$conf
    if ! cp $pitch_process_config $dir/conf/pitch_process.conf; then
      echo "$0: error copying pitch process config to $dir/conf/"
      exit 1;
    fi;
  fi

  echo "--silence-phones=$silphonelist" >>$conf
  echo "$0: created config file $conf"
fi

cp $srcdir/${iter}.mdl $dir/final.mdl || exit 1;
cp $srcdir/tree $dir/ || exit 1;
if [ -f $srcdir/frame_subsampling_factor ]; then
  cp $srcdir/frame_subsampling_factor $dir/
fi

if [ ! -z "$npfile" ]; then
  mkdir -p $dir/noise_vector/
  cp $npfile $dir/noise_vector || exit 1;
fi

> $dir/conf/nvector.conf
echo "--noise-prior=$dir/noise_vector/noise_prior" >>$dir/conf/nvector.conf
echo "--nvector-period=10" >>$dir/conf/nvector.conf
echo "--max-remembered-frames=1000" >> $dir/conf/nvector.conf
echo "--nvector-extraction-config=$dir/conf/nvector.conf" >>$conf


echo "$0: created config file $conf"
