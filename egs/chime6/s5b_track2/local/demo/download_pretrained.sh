#!/usr/bin/env bash
#
# Copyright  2021  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

# Begin configuration section.

# chime5 main directory path
chime5_corpus=/export/corpora5/CHiME5

# The following contains synchronized reecordings for CHiME-6
chime6_corpus=/export/c04/aarora8/kaldi/egs/chime6/s5c_track1/CHiME6
json_dir=${chime6_corpus}/transcriptions
audio_dir=${chime6_corpus}/audio

# Download pretrained SAD if not present
if [ ! -d exp/segmentation_1a ]; then
  echo "$0: Downloading CHiME-6 baseline SAD"
  wget -O 0012_sad_v1.tar.gz http://kaldi-asr.org/models/12/0012_sad_v1.tar.gz
  tar -xvzf 0012_sad_v1.tar.gz
  cp -r 0012_sad_v1/exp/segmentation_1a exp/
else
  echo "$0: SAD already present. Not downloading again."
fi

# Download i-vector and x-vector extractor if not present
if [ ! -d exp/xvector_nnet_1a ] || [ ! -d exp/vb_reseg/extractor_diag_c1024_i400 ]; then
  echo "$0: Downloading JHU-CLSP CHiME-6 i-vector and x-vector extractors"
  wget -O 0012_diarization_v2.tar.gz http://kaldi-asr.org/models/12/0012_diarization_v2.tar.gz
  tar -xvzf 0012_diarization_v2.tar.gz
  cp -r 0012_diarization_v2/exp/* exp/
else
  echo "$0: Diarization model already present. Not downloading again."
fi

# Download acoustic and language models if not present
if [ ! -d exp/rnnlm_lstm_1b ]; then
  echo "$0: Downloading JHU-CLSP CNN-TDNNF acoustic model and RNNLM"
  wget -O 0012_asr_v2.tar.gz http://kaldi-asr.org/models/12/0012_asr_v2.tar.gz
  tar -xvzf 0012_asr_v2.tar.gz
  cp -rL 0012_asr_v2/exp/* exp/
  else
  echo "$0: AM and RNNLM already present. Not downloading again."
fi

if [ ! -d pb_chime5/ ]; then
  echo "$0: Installing Paderborn's CHiME-5 GSS toolkit"
  local/install_pb_chime6.sh
  else
  echo "$0: Paderborn CHiME-5 GSS toolkit already installed."
fi

if [ ! -d pb_chime5/cache/CHiME6/transcriptions/dev ]; then
  (
  cd pb_chime5
  miniconda_dir=$HOME/miniconda3/
  export PATH=$miniconda_dir/bin:$PATH
  make cache/CHiME6 CHIME5_DIR=${chime5_corpus} CHIME6_DIR=${chime6_corpus}
  )
fi

exit
