#!/bin/bash
# This file creates one hot ivectors for each speaker.

stage=1
nnet3_affix=
ivector_dir=
spk2utt=

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

cmd='run.pl'

ivector_dim=$(< $spk2utt wc -l)

if [ $stage -le 1 ]; then
  echo "$0: Creating file to store one hot ivectors"
  [ -f ${ivector_dir}/ivectors_onehot.txt ] && \
      rm ${ivector_dir}/ivectors_onehot.txt
  touch ${ivector_dir}/ivectors_onehot.txt
fi

if [ $stage -le 2 ]; then
  echo "$0: Generating $ivector_dim one-hot ivectors"
  
  for i in `seq $ivector_dim`;do printf '[%*s]\n' $ivector_dim|tr ' ' '0'| \
    sed "s/0/1/$i"|sed -e 's/\(.\)/\1 /g';done > ${ivector_dir}/ivectors_onehot.txt

  cat ${ivector_dir}/ivectors_onehot.txt | \
      shuf > ${ivector_dir}/ivectors_onehot_shuffled.txt
  rm ${ivector_dir}/ivectors_onehot.txt
  mv ${ivector_dir}/ivectors_onehot_shuffled.txt ${ivector_dir}/ivectors_onehot.txt
fi

if [ $stage -le 3 ]; then
    echo "$0: Combining utt ids with corresponding ivectors"
    # Get list of speaker ids
    cut -d' ' -f2- ${spk2utt} > ${ivector_dir}/uttids
    touch ${ivector_dir}/ivectors_onehot_final.txt

    while read utts <&3 && read onehot <&4; do
      for utt in ${utts[@]}; do
        echo "${utt} ${onehot}" >> ${ivector_dir}/ivectors_onehot_final.txt
      done
    done 3<${ivector_dir}/uttids 4<${ivector_dir}/ivectors_onehot.txt

    rm ${ivector_dir}/uttids
    #rm ${ivector_dir}/ivectors_onehot.txt
    #mv ${ivector_dir}/ivectors_onehot_final.txt \
    #  ${ivector_dir}/ivectors_onehot.txt
    echo "$0: Creating scp and ark files"
    $cmd ${ivector_dir}/log/copy_vector.log \
      copy-vector ark:${ivector_dir}/ivectors_onehot_final.txt \
      ark,t,scp:${ivector_dir}/ivectors_onehot.ark,${ivector_dir}/ivector_online.scp
fi

