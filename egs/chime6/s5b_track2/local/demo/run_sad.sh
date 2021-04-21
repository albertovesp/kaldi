#!/usr/bin/env bash
#
# Copyright  2021  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

# Begin configuration section.
cmd=run.pl
nj=5
demo_out_dir=demo
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

nnet_type=stats
dir=exp/segmentation_1a
sad_work_dir=exp/sad_1a_${nnet_type}/
sad_nnet_dir=$dir/tdnn_${nnet_type}_sad_1a

test_dir=$1
data_name=$(basename $test_dir)

local/segmentation/detect_speech_activity.sh --nj $nj \
  --cmd "$cmd" \
  $test_dir $sad_nnet_dir mfcc $sad_work_dir \
  $test_dir || exit 1

vad_dir=${test_dir}_max_seg
steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
  ${vad_dir}/utt2spk ${vad_dir}/segments ${vad_dir}/rttm

cp ${test_dir}/wav.scp ${vad_dir}/  # needed for diarization

echo "$0: Scoring VAD output.."
ref_rttm=${test_dir}/ref_rttm
steps/segmentation/convert_utt2spk_and_segments_to_rttm.py ${test_dir}/utt2spk.bak \
  ${test_dir}/segments.bak ${test_dir}/ref_rttm

# To score, we select just U06 segments from the hypothesis RTTM.
hyp_rttm=${vad_dir}/rttm

sed 's/_U0[1-6].ENH//g' $ref_rttm | sort -u > $ref_rttm.scoring
sed 's/.ENH//g' $hyp_rttm > $hyp_rttm.scoring
cat ./local/uem_file | grep 'U06' | sed 's/_U0[1-6]//g' > ./local/uem_file.tmp
md-eval.pl -1 -c 0.25 -u ./local/uem_file.tmp -r $ref_rttm.scoring -s $hyp_rttm.scoring |\
  awk 'or(/MISSED SPEECH/,/FALARM SPEECH/)'

cat ${hyp_rttm}.scoring |\
  awk '{$8="speech"}{print $0}' |\
  local/demo/rttm2labels.py > ${demo_out_dir}/${data_name}.vad 

exit
