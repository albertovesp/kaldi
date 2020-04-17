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

# online_cmvn_config can be used both for nn-features and i-vector features.
# If the file $dir/online_cmvn exists, it is used for both feature streams.
# The $dir/online_cmvn 'flag' file is created when training with online-cmvn.
online_cmvn_config=conf/online_cmvn.conf

ivector_period=10 # Number of frames for which the i-vector stays the same
                  # (use same value as from local/nnet3/run_ivector_common.sh).
per_utt_basis=true # If true, then treat each utterance as a separate speaker
                   # for purposes of basis training... this is recommended if
                   # the number of actual speakers in your training set is less
                   # than (feature-dim) * (feature-dim+1).
silence_weight=0.01
iter=final
# End configuration.

cmd=run.pl

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# -ne 6 ] && [ $# -ne 3 ]; then
   echo "Usage: $0 [options] <data-dir> <lang-dir> <noise-prior-file> <nnet-dir> <gmm-dir> <output-dir>"
   echo "e.g.: $0 data/train data/lang exp/noise_prior exp/nnet3 exp/tri exp/nnet3_online"
   echo "main options (for others, see top of script file)"
   echo "  --feature-type <mfcc|plp>                        # Type of the base features; "
   echo "                                                   # important to generate the correct"
   echo "                                                   # configs in <output-dir>/conf/"
   echo "  --add-pitch <true|false>                         # Append pitch features to cmvn"
   echo "                                                   # (default: false)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --config <config-file>                           # config containing options"
   echo "  --per-utt-basis <true|false>                     # Do basis computation per utterance"
   echo "                                                   # (default: true)"
   echo "  --silence-weight <weight>                        # Weight on silence for basis fMLLR;"
   echo "                                                   # default 0.01."  
   echo "  --iter <model-iteration|final>                   # iteration of model to take."
   echo "  --stage <stage>                                  # stage to do partial re-run from."
   exit 1;
fi


if [ $# -eq 6 ]; then
  data=$1
  lang=$2
  npfile=$3
  nnet_srcdir=$4
  gmm_srcdir=$5
  dir=$6
fi

data_hires=${data}_sp_hires

for f in $lang/phones/silence.csl $nnet_srcdir/${iter}.mdl $nnet_srcdir/tree \
  $gmm_srcdir/final.mdl $gmm_srcdir/ali.1.gz $data/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
if [ ! -z "$npfile" ]; then
  [ ! -f $npfile ] && echo "$0: no such file $npfile" && exit 1;
fi


dir=$(utils/make_absolute.sh $dir) # Convert $dir to an absolute pathname, so that the
                        # configuration files we write will contain absolute
                        # pathnames.
mkdir -p $dir/conf
mkdir -p $dir/gmm
mkdir -p $dir/log

# First we add config options related to the GMM
gmm_dir=$dir/gmm

nj=`cat $gmm_srcdir/num_jobs` || exit 1;
sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

echo $nj >$dir/num_jobs || exit 1;

utils/lang/check_phones_compatible.sh $lang/phones.txt $gmm_srcdir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

splice_opts=`cat $gmm_srcdir/splice_opts 2>/dev/null`
cmvn_opts=`cat $gmm_srcdir/cmvn_opts 2>/dev/null`
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
cp $gmm_srcdir/splice_opts $gmm_srcdir/cmvn_opts $gmm_srcdir/final.mat $gmm_srcdir/final.mdl $gmm_dir/

rm $dir/{plp,mfcc,fbank}.conf 2>/dev/null
echo "$0: preparing configuration files in $dir/conf"

if [ -f $dir/conf/online.conf ]; then
  echo "$0: moving $dir/conf/online.conf to $dir/conf/online.conf.bak"
  mv $dir/conf/online.conf $dir/conf/online.conf.bak
fi

conf=$dir/conf/online.conf
echo -n >$conf

cp $online_cmvn_config $dir/conf/online_cmvn.conf || exit 1;

if $add_pitch; then
  echo "$0: enabling pitch features"
  echo "--add-pitch=true" >>$conf
  echo "--gmm.add-pitch=true" >>$conf
  echo "$0: creating $dir/conf/online_pitch.conf"
  if [ ! -f $online_pitch_config ]; then
    echo "$0: expected file '$online_pitch_config' to exist.";
    exit 1;
  fi
  cp $online_pitch_config $dir/conf/online_pitch.conf || exit 1;
  echo "--online-pitch-config=$dir/conf/online_pitch.conf" >>$conf
  echo "--gmm.online-pitch-config=$dir/conf/online_pitch.conf" >>$conf
fi

# create global_cmvn.stats
if ! matrix-sum --binary=false scp:$data/cmvn.scp - >$gmm_dir/global_cmvn.stats 2>/dev/null; then
  echo "$0: Error summing cmvn stats"
  exit 1
fi

# Set up the unadapted features "$sifeats".
if [ -f $gmm_dir/final.mat ]; then feat_type=lda; else feat_type=delta; fi

if $add_pitch; then
  skip_opt="--skip-dims=13:14:15" # should make this more general.
fi

echo "$0: feature type is $feat_type";
case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"
        online_sifeats="ark,s,cs:apply-cmvn-online $skip_opt --config=$online_cmvn_config $gmm_dir/global_cmvn.stats $online_cmvn_spk2utt_opt scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $gmm_dir/final.mat ark:- ark:- |"
       online_sifeats="ark,s,cs:apply-cmvn-online $skip_opt --config=$online_cmvn_config $online_cmvn_spk2utt_opt $gmm_dir/global_cmvn.stats scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $gmm_dir/final.mat ark:- ark:- |";;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

# Set up the adapted features "$feats" for training set.
if [ -f $gmm_srcdir/trans.1 ]; then
  feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$gmm_srcdir/trans.JOB ark:- ark:- |";
else
  feats="$sifeats";
fi


if $per_utt_basis; then
  spk2utt_opt=  # treat each utterance as separate speaker when computing basis.
  echo "Doing per-utterance adaptation for purposes of computing the basis."
else
  echo "Doing per-speaker adaptation for purposes of computing the basis."
  [ `cat $sdata/spk2utt | wc -l` -lt $[41*40] ] && \
    echo "Warning: number of speakers is small, might be better to use --per-utt=true."
  spk2utt_opt="--spk2utt=ark:$sdata/JOB/spk2utt"
fi

if [ $stage -le 0 ]; then
  echo "$0: Accumulating statistics for basis-fMLLR computation"
# Note: we get Gaussian level alignments with the "final.mdl" and the
# speaker adapted features.
  $cmd JOB=1:$nj $dir/log/basis_acc.JOB.log \
    ali-to-post "ark:gunzip -c $gmm_srcdir/ali.JOB.gz|" ark:- \| \
    weight-silence-post $silence_weight $silphonelist $gmm_dir/final.mdl ark:- ark:- \| \
    gmm-post-to-gpost $gmm_dir/final.mdl "$feats" ark:- ark:- \| \
    gmm-basis-fmllr-accs-gpost $spk2utt_opt \
    $gmm_dir/final.mdl "$sifeats" ark,s,cs:- $gmm_dir/basis.acc.JOB || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: computing the basis matrices."
  $cmd $dir/log/basis_training.log \
    gmm-basis-fmllr-training $gmm_dir/final.mdl $gmm_dir/fmllr.basis $gmm_dir/basis.acc.* || exit 1;
  if $cleanup; then
    rm $gmm_dir/basis.acc.* 2>/dev/null
  fi
fi

if [ $stage -le 2 ]; then
  echo "$0: accumulating stats for online alignment model."

  # Accumulate stats for "online alignment model"-- this model is computed with
  # the speaker-independent features and online CMVN, but matches
  # Gaussian-for-Gaussian with the final speaker-adapted model.

  $cmd JOB=1:$nj $dir/log/acc_alimdl.JOB.log \
    ali-to-post "ark:gunzip -c $gmm_srcdir/ali.JOB.gz|" ark:-  \| \
    gmm-acc-stats-twofeats $gmm_dir/final.mdl "$feats" "$online_sifeats" \
    ark,s,cs:- $gmm_dir/final.JOB.acc || exit 1;
  [ `ls $gmm_dir/final.*.acc | wc -w` -ne "$nj" ] && echo "$0: Wrong #accs" && exit 1;
  # Update model.
  $cmd $dir/log/est_online_alimdl.log \
    gmm-est --remove-low-count-gaussians=false $gmm_dir/final.mdl \
    "gmm-sum-accs - $gmm_dir/final.*.acc|" $gmm_dir/final.oalimdl  || exit 1;
  if $cleanup; then
    rm $gmm_dir/final.*.acc
  fi
fi

if [ $stage -le 3 ]; then
  case "$feature_type" in
    mfcc)
      echo "$0: creating $dir/conf/mfcc.conf"
      echo "--gmm.mfcc-config=$dir/conf/mfcc.conf" >>$conf
      cp conf/mfcc.conf $dir/conf/
      echo "$0: creating $dir/conf/mfcc_hires.conf"
      echo "--mfcc-hires-config=$dir/conf/mfcc_hires.conf" >>$conf
      cp conf/mfcc_hires.conf $dir/conf/ ;;
    *)
      echo "Unknown feature type $feature_type"
  esac
  if ! cp $online_cmvn_config $dir/conf/online_cmvn.conf; then
    echo "$0: error copying online cmvn config to $dir/conf/"
    exit 1;
  fi
  echo "--cmvn-config=$dir/conf/online_cmvn.conf" >>$conf
  echo "--gmm.cmvn-config=$dir/conf/online_cmvn.conf" >>$conf
  if [ -f $gmm_dir/final.mat ]; then
    echo "$0: enabling feature splicing"
    echo "--gmm.splice-feats" >>$conf
    echo "$0: creating $dir/conf/splice.conf"
    for x in $(cat $gmm_dir/splice_opts); do echo $x; done > $dir/conf/splice.conf
    echo "--gmm.splice-config=$dir/conf/splice.conf" >>$conf
    echo "$0: enabling LDA"
    echo "--gmm.lda-matrix=$gmm_dir/final.mat" >>$conf
  else
    echo "$0: enabling deltas"
    echo "--add-deltas" >>$conf
    echo "--gmm.add-deltas" >>$conf
  fi
  if $add_pitch; then
    echo "$0: enabling pitch features"
    echo "--add-pitch" >>$conf
    echo "--gmm.add-pitch" >>$conf
    echo "$0: creating $dir/conf/pitch.conf"
    echo "--pitch-config=$dir/conf/pitch.conf" >>$conf
    echo "--gmm.pitch-config=$dir/conf/pitch.conf" >>$conf
    if ! cp $pitch_config $dir/conf/pitch.conf; then
      echo "$0: error copying pitch config to $dir/conf/"
      exit 1;
    fi;
    echo "$0: creating $dir/conf/pitch_process.conf"
    echo "--pitch-process-config=$dir/conf/pitch_process.conf" >>$conf
    echo "--gmm.pitch-process-config=$dir/conf/pitch_process.conf" >>$conf
    if ! cp $pitch_process_config $dir/conf/pitch_process.conf; then
      echo "$0: error copying pitch process config to $dir/conf/"
      exit 1;
    fi;
    nfields=$(sed -n '2,2p' $gmm_dir/global_cmvn.stats | \
      perl -e '$_ = <>; s/^\s+|\s+$//g; print scalar(split);');
    if [ $nfields != 17 ]; then
      echo "$0: $gmm_dir/global_cmvn.stats has $nfields entries per row (expected 17)."
      echo "$0: Did you append pitch features?"
      exit 1;
    fi
  fi

  echo "--gmm.fmllr-basis=$gmm_dir/fmllr.basis" >>$conf
  echo "--gmm.online-alignment-model=$gmm_dir/final.oalimdl" >>$conf
  echo "--gmm.model=$gmm_dir/final.mdl" >>$conf
  echo "--silence-phones=$silphonelist" >>$conf
  echo "--gmm.silence-phones=$silphonelist" >>$conf
  echo "--global-cmvn-stats=$gmm_dir/global_cmvn.stats" >>$conf
  echo "--gmm.global-cmvn-stats=$gmm_dir/global_cmvn.stats" >>$conf
  echo "$0: created config file $conf"
fi

cp $nnet_srcdir/${iter}.mdl $dir/final.mdl || exit 1;
cp $nnet_srcdir/tree $dir/ || exit 1;
if [ -f $nnet_srcdir/frame_subsampling_factor ]; then
  cp $nnet_srcdir/frame_subsampling_factor $dir/
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
