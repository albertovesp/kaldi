#!/usr/bin/env bash
#
# This scripts provides inference code for a single wav file.
# ICASSP 2021 CHiME-6 demo.
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe and Yenda Trmal)
#            2019  Desh Raj, David Snyder, Ashish Arora
# Apache 2.0

# How to run:
# This script can be run one stage at a time by providing the `--stage-name`
# argument in (prep, sad, diarize, gss, asr, rnnlm), or by providing the `--stage`
# argument as in Kaldi scripts. The stage-name descriptions are:
# 1. `--stage-name prep`: Downloads pre-trained models and prepares data directory
#     for the session.
# 2. `--stage-name sad`: Applies SAD on the wav file and prints error rates
# 3. `--stage-name diarize`: Applies overlap-aware VB resegmentation
# 4. `--stage-name gss`: Performs multi-array GSS-based enhancement
# 5. `--stage-name asr`: Decodes GSS-enhanced segments using the CNN-TDNNF acoustic model
# 6. `--stage-name rnnlm`: Performs RNNLM rescoring
#
# Example usage:
# ./run_demo.sh --stage-name prep
# Note: We use the session S02 in the dev set for this demo. To save time, we
# will directly start with WPE+Beamformed files.

# Begin configuration section.
stage=0
stage_name=prep  # prep, sad, diarize, gss, asr, rnnlm

#========================================================================================
# How to run this tutorial (step-by-step)

# Step 1: ./run_demo.sh --stage-name prep --set dev --session S02 
set=dev
session=S02

# Step 2: ./run_demo.sh --stage-name sad --test-dir data/S02_beamformit_dereverb 
test_dir=data/${session}_beamformit_dereverb

# Step 3: ./run_demo.sh --stage-name diarize --seg-dir data/S02_beamformit_dereverb_max_seg
seg_dir=${test_dir}_max_seg
diar_stage=1
# When the process is at "Detecting overlaps...", stop and uncomment the following:
# cd exp && ln -s /export/c03/draj/slt/kaldi/egs/chime6/s5b_track2/exp/S02_beamformit_dereverb_max_seg_ovl . && cd .. || exit 1

# And then run again with modified diar-stage:
# Step 3b: ./run_demo.sh --stage-name diarize --seg-dir data/S02_beamformit_dereverb_max_seg --diar-stage 6

# Step 4: ./run_demo.sh --stage-name gss --context-samples 320000 --iterations 5 --rttm-file exp/S02_beamformit_dereverb_max_seg_vb/Vb_rttm_ol
context_samples=320000
iterations=5
seg_data_name=$(basename $seg_dir)
rttm_file=exp/${seg_data_name}_vb/Vb_rttm_ol
# Run the following to use pre-computed GSS enhanced wavs.
# ln -s /export/c03/draj/slt/kaldi/egs/chime6/s5b_track2/gss_cs320000_it5 .

# Step 5: ./run_demo.sh --stage-name asr --gss-enhanced-dir gss_cs320000_it5
gss_enhanced_dir=gss_cs${context_samples}_it${iterations}

# Step 6: ./run_demo.sh --stage-name score

#=========================================================================================
# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh

# chime5 main directory path
chime5_corpus=/export/corpora5/CHiME5

# The following contains synchronized reecordings for CHiME-6
chime6_corpus=/export/c04/aarora8/kaldi/egs/chime6/s5c_track1/CHiME6
json_dir=${chime6_corpus}/transcriptions
audio_dir=${chime6_corpus}/audio

# The following contains WPE+beamformed recordings
enhanced_dir=/export/c03/draj/kaldi_jsalt/egs/chime6/s5b_track2/enhanced

# The following will store outputs (from intermediate stages)
demo_out_dir=demo

set -e # exit on error

if [ $stage -le -1 ]; then
  echo "$0: "
fi

mkdir -p ${demo_out_dir}

# This stage downloads all pretrained models required for the inference
# pipeline below. It also prepares the data directory for the session
# and extracts 40-dim MFCCs for SAD.
if [[ ( $stage -le 0 ) && ( $stage_name == 'prep') ]]; then
  echo "$0: Downloading and extracting pretrained models for inference"
  local/demo/download_pretrained.sh

  echo "$0: Preparing data directory for ${session}"
  local/demo/prepare_data.sh --session ${session} \
    "${enhanced_dir}/${set}_beamformit_u0*" \
    ${json_dir}/${set} data/${session}_beamformit_dereverb

  echo "$0: Extracting features (40-dim MFCCs)"
  steps/make_mfcc.sh --nj 5 --cmd "$demo_cmd" \
    --mfcc-config conf/mfcc_hires.conf \
    data/${session}_beamformit_dereverb

  echo "$0: Data directory prepared at data/${session}_beamformit_dereverb"
fi

# The next stage applies the pretrained VAD. It applies the VAD independently
# on all arrays in the session and then does posterior fusion across all arrays.
if [[ ( $stage -le 1 ) && ( $stage_name == 'sad') ]]; then
  echo "$0: Performing speech activity detection on ${test_dir}"
  local/demo/run_sad.sh --nj 5 --cmd "$demo_cmd" \
    ${test_dir}
  # MISSED SPEECH =    114.76 secs (  1.8 percent of scored time)
  # FALARM SPEECH =     45.22 secs (  0.7 percent of scored time)
  echo "$0: Segmented directory is at ${test_dir}_max_seg"
fi

# The next stage performs diarization using overlap-aware VB resegmentation.
if [[ ( $stage -le 2 ) && ( $stage_name == 'diarize') ]]; then
  echo "$0: Performing speech activity detection on ${seg_dir}"
  local/demo/run_diarize.sh --nj 5 --cmd "$demo_cmd" --session ${session} \
    --ref-rttm ${test_dir}/ref_rttm --stage $diar_stage \
    $seg_dir
  
  # DER: 54.81, JER: 66.40
fi

# The next stage performs GSS-based enhancement.
if [[ ( $stage -le 3 ) && ( $stage_name == 'gss') ]]; then
  echo "$0: Performing GSS using ${rttm_file}"
  ./local/demo/run_gss.sh --set ${set} --session ${session} \
    --context-samples ${context_samples} --iterations ${iterations} \
    ${rttm_file}

  echo "$0: Enhanced wav files saved to gss_cs${context_samples}_it${iterations}/audio/${set}"
fi

if [[ ( $stage -le 4 ) && ( $stage_name == 'asr') ]]; then
  echo "$0: Performing ASR decoding on wav files in ${gss_enhanced_dir}"  
  local/demo/run_asr.sh --session ${session} --set ${set} --cmd "$demo_cmd" \
    $gss_enhanced_dir
fi

exit 1
#######################################################################
# Score decoded dev/eval sets
#######################################################################
if [ $stage -le 8 ]; then
  # final scoring to get the challenge result
  # please specify both dev and eval set directories so that the search parameters
  # (insertion penalty and language model weight) will be tuned using the dev set
  local/score_for_submit.sh --stage $score_stage \
      --dev_decodedir exp/chain_${train_set}_cleaned_rvb/tdnn1b_cnn_sp/decode_dev_beamformit_dereverb_diarized_2stage \
      --dev_datadir dev_beamformit_dereverb_diarized_hires \
      --eval_decodedir exp/chain_${train_set}_cleaned_rvb/tdnn1b_sp/decode_eval_beamformit_dereverb_diarized_2stage \
      --eval_datadir eval_beamformit_dereverb_diarized_hires
fi
exit 0;
