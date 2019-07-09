#!/bin/bash

stage=0
db_dir=/export/corpora/ASTAR/RedDots
data_dir=data/train
vad_dir=$data_dir/vad

ivector_dim=768
xvector_dim=512
ivector_extractor=exp/extractor_${ivector_dim}
xvector_extractor=exp/xvector_nnet_${xvector_dim}_aug

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -euo pipefail

if [ $stage -le 0 ]; then
  mkdir -p $data_dir
  local/prepare_data.py $db_dir $data_dir
  filter_scp.pl $data_dir/utt2spk $data_dir/text > $data_dir/utt2txt
fi

if [ $stage -le 1 ]; then
  echo "$0: Extracting MFCC features for all data..."
  utils/fix_data_dir.sh $data_dir
  steps/make_mfcc.sh --write-utt2num-frames true \
    --cmd "$train_cmd" --nj 16 $data_dir
  steps/compute_cmvn_stats.sh $data_dir
  utils/fix_data_dir.sh $data_dir
  utils/data/get_utt2dur.sh $data_dir
  utils/validate_data_dir.sh $data_dir
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    $data_dir exp/make_vad $vad_dir
fi

if [ $stage -le 2 ]; then
  # In this stage, we prepare augmented utterances by adding
  # music, noise, and babble. This increases the number of
  # utterances per speaker from 3 to 12.
  musan_root=/export/corpora/JHU/musan
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train/utt2num_frames > data/train/reco2dur
  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  steps/data/make_musan.sh --sampling-rate 8000 $musan_root data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train data/train_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train data/train_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train data/train_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh --extra-files "utt2gender utt2txt utt2dur utt2num_frames" data/train_aug data/train data/train_noise data/train_music data/train_babble
fi

data_dir=data/train_aug

if [ $stage -le 3 ]; then
  echo "$0: Extracting MFCC features for all data..."
  utils/fix_data_dir.sh $data_dir
  steps/make_mfcc.sh --cmd "$train_cmd" \
    --nj 16 $data_dir
  steps/compute_cmvn_stats.sh $data_dir
  utils/fix_data_dir.sh $data_dir
  utils/data/get_utt2dur.sh $data_dir
  utils/validate_data_dir.sh $data_dir
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    $data_dir exp/make_vad $vad_dir
fi

if [ $stage -le 4 ]; then
  echo "$0: extracting hires MFCC features for all data..."
  utils/copy_data_dir.sh $data_dir ${data_dir}_hires
  steps/make_mfcc.sh --nj 30 --mfcc-config conf/mfcc_hires.conf \
    --cmd "$train_cmd" ${data_dir}_hires || exit 1;
  steps/compute_cmvn_stats.sh ${data_dir}_hires || exit 1;
  utils/fix_data_dir.sh ${data_dir}_hires
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    ${data_dir}_hires exp/make_vad ${data_dir}_hires
fi

if [ $stage -le 5 ]; then
  echo "$0: extracting i-vectors for all utterances"
  local/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 80 \
    $ivector_extractor ${data_dir}_hires exp/ivectors_${ivector_dim}
fi

if [ $stage -le 6 ]; then
  echo "$0: extracting x-vectors for all utterances"
  local/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 80 \
      $xvector_extractor ${data_dir}_hires exp/xvectors_${xvector_dim}_aug
fi

# Now we have each stage corresponding to a different analysis
# To add more, copy scripts and make appropriate changes.
# Note: Each of these require Pytorch and kaldi_io
if [ $stage -le 7 ]; then
  # Session id classification
  output_file=RESULTS_SESSID
  touch $output_file
  #local/probe_spkid.py exp/ivectors_${ivector_dim}/ivector.scp ${data_dir}/utt2spk \
  #  --feat_dim $ivector_dim --num_epochs 50 --output_file $output_file
  
  local/probe_spkid.py exp/xvectors_${xvector_dim}_aug/xvector_mat.scp ${data_dir}_hires/utt2spk \
    --feat_dim $xvector_dim --num_epochs 50 --output_file $output_file
fi

if [ $stage -le 8 ]; then
  # Gender classification
  output_file=RESULTS_GENDER
  touch $output_file
  #local/probe_gender.py exp/ivectors_${ivector_dim}/ivector.scp ${data_dir}/utt2gender \
  #  --feat_dim $ivector_dim --num_epochs 5 --output_file $output_file
  
  local/probe_gender.py exp/xvectors_${xvector_dim}_aug/xvector_mat.scp ${data_dir}_hires/utt2gender \
    --feat_dim $xvector_dim --num_epochs 20 --output_file $output_file
fi

if [ $stage -le 9 ]; then
  # Utterance length classification
  output_file=RESULTS_DUR
  touch $output_file
  #local/probe_dur.py exp/ivectors_${ivector_dim}/ivector.scp ${data_dir}/utt2dur \
  #  --feat_dim $ivector_dim --num_epochs 40 --output_file $output_file \
  #  --np_array_file dur_ivec_${ivector_dim}
  
  local/probe_dur.py exp/xvectors_${xvector_dim}_aug/xvector_mat.scp ${data_dir}_hires/utt2dur \
    --feat_dim $xvector_dim --num_epochs 20 --output_file $output_file \
    --np_array_file dur_xvec_${xvector_dim}
fi

if [ $stage -le 10 ]; then
  # Speaking rate classification
  # 3-way speed perturbation (0.5, 1.0, 1.5)
  # In this stage we prepare the speed-perturbed data and the vectors
  #local/perturb_data_dir_speed_3way.sh --always-include-prefix true \
  #  ${data_dir} ${data_dir}_sp

  #local/perturb_data_dir_speed_3way.sh --always-include-prefix true \
  #  ${data_dir}_hires ${data_dir}_sp_hires

  #echo "$0: Extracting MFCC features for speed-perturbed data..."
  #utils/fix_data_dir.sh $data_dir
  #steps/make_mfcc.sh --cmd "$train_cmd" \
  #  --nj 16 ${data_dir}_sp
  #steps/compute_cmvn_stats.sh ${data_dir}_sp
  #utils/fix_data_dir.sh ${data_dir}_sp
  #utils/data/get_utt2dur.sh ${data_dir}_sp
  #utils/validate_data_dir.sh ${data_dir}_sp
  #sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
  #  ${data_dir}_sp exp/make_vad ${vad_dir}_sp

  #echo "$0: extracting hires MFCC features for speed-perturbed data..."
  #utils/copy_data_dir.sh ${data_dir}_sp ${data_dir}_sp_hires
  #steps/make_mfcc.sh --nj 30 --mfcc-config conf/mfcc_hires.conf \
  #  --cmd "$train_cmd" ${data_dir}_sp_hires || exit 1;
  #steps/compute_cmvn_stats.sh ${data_dir}_sp_hires || exit 1;
  #utils/fix_data_dir.sh ${data_dir}_sp_hires
  #sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
  #  ${data_dir}_sp_hires exp/make_vad ${data_dir}_sp_hires

  #echo "$0: extracting i-vectors for speed-perturbed utterances"
  #local/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 80 \
  #  $ivector_extractor ${data_dir}_sp_hires exp/ivectors_sp_${ivector_dim}
  
  echo "$0: extracting x-vectors for speed-perturbed utterances"
  local/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 80 \
      $xvector_extractor ${data_dir}_sp_hires exp/xvectors_sp_${xvector_dim}_aug
  
fi

if [ $stage -le 11 ]; then
  output_file=RESULTS_SPEED
  touch $output_file
  #local/probe_speed.py exp/ivectors_sp_${ivector_dim}/ivector.scp \
  #  --feat_dim $ivector_dim --num_epochs 20 --output_file $output_file \
  #  --test_split 0.2
  
  local/probe_speed.py exp/xvectors_sp_${xvector_dim}_aug/xvector_mat.scp \
    --feat_dim $xvector_dim --num_epochs 10 --output_file $output_file \
    --test_split 0.2

fi

if [ $stage -le 12 ]; then
  # Augmentation type classification
  output_file=RESULTS_AUG
  touch $output_file
  #local/probe_aug_type.py exp/ivectors_${ivector_dim}/ivector.scp \
  #  --feat_dim $ivector_dim --num_epochs 50 --output_file $output_file 
  
  local/probe_aug_type.py exp/xvectors_${xvector_dim}_aug/xvector_mat.scp \
    --feat_dim $xvector_dim --num_epochs 40 --output_file $output_file 
fi

if [ $stage -le 13 ]; then
  # Utterance text classification
  output_file=RESULTS_TEXT
  touch $output_file
  #local/probe_text.py exp/ivectors_${ivector_dim}/ivector.scp ${data_dir}/utt2txt \
  #  --feat_dim $ivector_dim --num_epochs 100 --output_file $output_file
  
  local/probe_text.py exp/xvectors_${xvector_dim}_aug/xvector_mat.scp ${data_dir}_hires/utt2txt \
    --feat_dim $xvector_dim --num_epochs 100 --output_file $output_file
fi

if [ $stage -le 14 ]; then
  # Utterance word-level classification
  output_file=RESULTS_WORD
  touch $output_file
  #local/probe_word.py exp/ivectors_${ivector_dim}/ivector.scp ${data_dir}/utt2txt \
  #  --feat_dim $ivector_dim --num_epochs 5 --output_file $output_file --top_n 50
  
  local/probe_word.py exp/xvectors_${xvector_dim}_aug/xvector_mat.scp ${data_dir}_hires/utt2txt \
    --feat_dim $xvector_dim --num_epochs 5 --output_file $output_file --top_n 50
fi

if [ $stage -le 15 ]; then
  # Utterance phone-level classification
  output_file=RESULTS_PHONE
  touch $output_file
  #local/probe_phones.py exp/ivectors_${ivector_dim}/ivector.scp ${data_dir}/utt2txt \
  #  --feat_dim $ivector_dim --num_epochs 5 --output_file $output_file --top_n 50
  
  local/probe_phones.py exp/xvectors_${xvector_dim}_aug/xvector_mat.scp ${data_dir}_hires/utt2txt \
    --feat_dim $xvector_dim --num_epochs 5 --output_file $output_file --top_n 50
fi
