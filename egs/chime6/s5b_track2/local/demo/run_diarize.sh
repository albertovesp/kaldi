#!/usr/bin/env bash
#
# Copyright  2021  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

# Begin configuration section.
cmd=run.pl
nj=5
stage=1
session=S02
demo_out_dir=demo
ref_rttm=
# End configuration section
. ./utils/parse_options.sh  # accept options.. you can run this run.sh with the

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 1 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <data-dir>"
  echo -e >&2 "eg:\n  $0 data/train"
  exit 1
fi

set -e -o pipefail

seg_dir=$1
data_name=$(basename $seg_dir)

if [ ! -d ${seg_dir}/feats.scp ]; then
  echo "$0: Preparing VAD output for diarization"
  
  > ${seg_dir}/segments_all
  > ${seg_dir}/utt2spk_all
  cat ${seg_dir}/wav.scp | grep "U06" > ${seg_dir}/wav.scp
  cat ${seg_dir}/segments |\
    sed "s/${session}/${session}_U06/g" >> ${seg_dir}/segments.new
  mv ${seg_dir}/segments ${seg_dir}/segments.bak
  mv ${seg_dir}/segments.new ${seg_dir}/segments

  # Also prepare utt2spk and spk2utt files
  mv ${seg_dir}/utt2spk ${seg_dir}/utt2spk.bak
  cut -d' ' -f1,2 ${seg_dir}/segments > ${seg_dir}/utt2spk
  utils/utt2spk_to_spk2utt.pl ${seg_dir}/utt2spk > ${seg_dir}/spk2utt

  # Prepare new feats.scp
  steps/make_mfcc.sh --nj $nj --cmd "$cmd" \
    --mfcc-config conf/mfcc_hires.conf ${seg_dir}
fi

echo "$0: Applying diarization on ${session}"

local/diarize_vb.sh --nj 1 --cmd "$cmd" --ref-rttm $ref_rttm \
  --stage ${stage} \
  exp/xvector_nnet_1a \
  ${seg_dir} exp/${data_name}_diarization


cat exp/${data_name}_vb/VB_rttm_ol.scoring |\
  awk '($8=="5"){$8="1"}{print $0}' |\
  local/demo/rttm2labels.py > ${demo_out_dir}/${session}.diar 
echo "$0: Saved audacity labels to ${demo_out_dir}/${session}.diar"
exit
