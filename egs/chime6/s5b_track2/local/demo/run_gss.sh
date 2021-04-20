#!/usr/bin/env bash
#
# Copyright  2021  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

# Begin configuration section.
cmd=run.pl
nj=5
set=dev
session=S02
demo_out_dir=demo
context_samples=320000
iterations=5
ref_array_gss=U01
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

rttm_file=$1

gss_enhanced_dir=gss_cs${context_samples}_it${iterations}
mkdir -p ${gss_enhanced_dir}

set -e -o pipefail

echo "$0: Preparing diarized output for GSS"
rm pb_chime5/cache/CHiME6/transcriptions/*/*.json
cat ${rttm_file} |\
  sed "s/.ENH//g" |\
  local/truncate_rttm.py --min-segment-length 0.2 \
    - local/uem_file - |\
  sed "s/U06/${ref_array_gss}/g" |\
  awk '($8=="5"){$8="1"}{print $0}' |\
  local/convert_rttm_to_json.py - pb_chime5/cache/CHiME6/transcriptions/${set}/${session}.json

pushd pb_chime5
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
$HOME/miniconda3/bin/python -m pb_chime5.database.chime5.create_json -j cache/chime6.json -db cache/CHiME6 --transcription-path cache/CHiME6/transcriptions --chime6
popd

echo "$0: Performing GSS-based ennhancement on ${session}"
local/run_gss.sh \
  --cmd "$train_cmd --max-jobs-run 80" --nj 100 \
  --bss_iterations $iterations \
  --context_samples $context_samples \
  ${set} \
  ${gss_enhanced_dir} \
  ${gss_enhanced_dir} || exit 1

echo "$0: Renaming GSS ouput wav files for decoding"
for spk in `seq 1 4`; do
  find ${gss_enhanced_dir}/audio/${set} -name *.wav -exec \
    rename "s/${spk}_${session}/${session}_U06.ENH-${spk}/" {} \;
done

exit
