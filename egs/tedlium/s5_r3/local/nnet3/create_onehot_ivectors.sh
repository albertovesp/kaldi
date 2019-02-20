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
echo "The ivector dimension is ${ivector_dim}"

# Create file
if [ $stage -le 1 ]; then
    [ -f ${ivector_dir}/ivectors_onehot.txt ] && \
      rm ${ivector_dir}/ivectors_onehot.txt
    touch ${ivector_dir}/ivectors_onehot.txt
fi

# Create one-hot vectors
# TODO: this can be optimized
if [ $stage -le 2 ]; then
  for i in `seq ${ivector_dim}`; do
    echo -n '[ ' >> ${ivector_dir}/ivectors_onehot.txt
    for j in `seq ${ivector_dim}`; do
        echo -n $((i==j)) ' ' >> ${ivector_dir}/ivectors_onehot.txt
    done
    echo ']' >> ${ivector_dir}/ivectors_onehot.txt
  done
  
  cat ${ivector_dir}/ivectors_onehot.txt | \
      awk 'BEGIN{srand();}{print rand()"\t"$0}' | \
      sort -k1 -n | \
      cut -f2- > ${ivector_dir}/ivectors_onehot.txt
fi

if [ $stage -le 3 ]; then
    # Get list of speaker ids
    cut -d' ' -f2- ${spk2utt} > ${ivector_dir}/uttids
    touch ${ivector_dir}/ivectors_onehot_final.txt

    while read utts <&3 && read onehot <&4; do
      for utt in ${utts[@]}; do
        echo "${utt} ${onehot}" >> ${ivector_dir}/ivectors_onehot_final.txt
      done
    done 3<${ivector_dir}/uttids 4<${ivector_dir}/ivectors_onehot.txt

    rm ${ivector_dir}/uttids
    rm ${ivector_dir}/ivectors_onehot.txt
    mv ${ivector_dir}/ivectors_onehot_final.txt \
      ${ivector_dir}/ivectors_onehot.txt

    $cmd ${ivector_dir}/log/copy_vector.log \
      copy-vector ark:${ivector_dir}/ivectors_onehot.txt \
      ark,t,scp:${ivector_dir}/ivectors_onehot.ark,${ivector_dir}/ivectors_onehot.scp
fi

