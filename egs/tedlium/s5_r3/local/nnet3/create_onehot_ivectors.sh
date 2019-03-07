#!/bin/bash
# This file creates one hot ivectors for each speaker.

stage=
nnet3_affix=
ivector_dir=
spk2utt=
online_ivector_dir=

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

cmd = 'queue.pl'

ivector_dim=$(< $spk2utt wc -l)

if [ $stage -le 1 ]; then
  echo "$0: Creating initial i-vectors matrix"

  ivector-speaker-matrix --bias=false ark:$spk2utt \
    scp:${online_ivector_dir}/ivectors_utt.scp \
    ${ivector_dir}/initial_ivectors.mat
fi

if [ $stage -le 2 ]; then
  echo "$0: Creating file to store one hot ivectors"
  [ -f ${ivector_dir}/ivectors_onehot.txt ] && \
      rm ${ivector_dir}/ivectors_onehot.txt
  touch ${ivector_dir}/ivectors_onehot.txt
  
  echo "$0: Generating $ivector_dim one-hot ivectors"
  
  for i in `seq $ivector_dim`;do printf '[%*s]\n' $ivector_dim|tr ' ' '0'| \
    sed "s/0/1/$i"|sed -e 's/\(.\)/\1 /g';done > ${ivector_dir}/ivectors_onehot.txt

fi

if [ $stage -le 3 ]; then
    echo "$0: Combining utt ids with corresponding ivectors"
    # Get list of speaker ids
    cut -d' ' -f2- ${spk2utt} > ${ivector_dir}/uttids
    [ -f ${ivector_dir}/ivectors_onehot_final.txt ] && \
      rm ${ivector_dir}/ivectors_onehot_final.txt
    touch ${ivector_dir}/ivectors_onehot_final.txt

    while read utts <&3 && read onehot <&4; do
      for utt in ${utts[@]}; do
        echo "${utt} ${onehot}" >> ${ivector_dir}/ivectors_onehot_final.txt
      done
    done 3<${ivector_dir}/uttids 4<${ivector_dir}/ivectors_onehot.txt

    rm ${ivector_dir}/uttids
    echo "$0: Creating scp and ark files"
    copy-vector ark:${ivector_dir}/ivectors_onehot_final.txt \
      ark,t,scp:${ivector_dir}/ivectors_onehot.ark,${ivector_dir}/ivector_online.scp
fi
exit 0
