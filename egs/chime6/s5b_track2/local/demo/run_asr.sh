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
asr_model_dir=exp/chain_train_worn_simu_u400k_cleaned_rvb
ivector_dir=exp/nnet3_train_worn_simu_u400k_cleaned_rvb
# End configuration section
. ./utils/parse_options.sh  # accept options.. you can run this run.sh with the

. ./path.sh

if [ $# -ne 1 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <data-dir>"
  echo -e >&2 "eg:\n  $0 data/train"
  exit 1
fi

gss_enhanced_dir=$1

set -e -o pipefail

echo "$0: Preparing data dir with GSS-enhanced wav files for decoding"

data_dir=data/${session}_diarized
mkdir -p $data_dir

find -L ${gss_enhanced_dir}/audio/${set} -name  "S[0-9]*.wav" | \
  perl -ne '{
    chomp;
    $path = $_;
    next unless $path;
    @F = split "/", $path;
    ($f = $F[@F-1]) =~ s/.wav//;
    print "$f $path\n";
  }' | sort > $data_dir/wav.scp

paste <(cut -d' ' -f1 $data_dir/wav.scp) <(cut -d'-' -f1,2 $data_dir/wav.scp) \
  >$data_dir/utt2spk
utils/utt2spk_to_spk2utt.pl $data_dir/utt2spk > $data_dir/spk2utt
utils/fix_data_dir.sh ${data_dir}
utils/validate_data_dir.sh --no-text --no-feats $data_dir || exit 1

echo "$0 Performing decoding on the enhanced files"

local/nnet3/decode.sh --affix 2stage --acwt 1.0 --post-decode-acwt 10.0 \
  --frames-per-chunk 150 --nj 4 --ivector-dir ${ivector_dir} \
  data/${session}_diarized data/lang $asr_model_dir/tree_sp/graph $asr_model_dir/tdnn1b_cnn_sp/

exit
