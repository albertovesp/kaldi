#!/bin/bash
echo "Working on machine $HOSTNAME"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/NVIDIA/cuda-9.0/lib64

data=
clean_data=
nnet_dir=
egs_dir=$nnet_dir/egs
stage=11
train_stage=-1
feat_dim=40
num_classes=7325

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh


# Training config options
seed=7
train_num_egs=14
valid_num_egs=3
epochs=3
batch_size=128
loss=ce
metric=none
optimizer=sgd
lr=0.01
arch=tdnn
ngpu=1
num_workers=6
use_relu=0


#exp_name=debug
exp_name=arch${arch}epoch${epochs}lr${lr}bs${batch_size}loss${loss}metric${metric}op${optimizer}relu${use_relu}seed${seed}
mkdir -p $nnet_dir/log
mkdir -p $nnet_dir/model
mkdir -p $egs_dir

if [ $stage -le 11 ] & [ ! -z $clean_data ]; then
  # Create a scp file containing RIR utterance ids and corresponding
  # clean utterance MFCC path.
  rir_uttid=$data/rir_id
  clean_uttid=$data/clean_id
  cut -d " " -f1 $data/wav.scp > $rir_uttid
  cut -d "-" -f4- $rir_uttid > $clean_uttid
  paste $clean_uttid $rir_uttid | sort -k1 | join - $clean_data/feats.scp -a1 |\
    cut -d " " -f 2,3 | sort -k1 > $data/clean_feats.scp
  rm $rir_uttid
  rm $clean_uttid 
fi

num_pdfs=$(awk '{print $2}' $data/utt2spk | sort | uniq -c | wc -l)

# Now we create the nnet examples using sid/nnet3/xvector/get_egs.sh.
# The argument --num-repeats is related to the number of times a speaker
# repeats per archive.  If it seems like you're getting too many archives
# (e.g., more than 200) try increasing the --frames-per-iter option.  The
# arguments --min-frames-per-chunk and --max-frames-per-chunk specify the
# minimum and maximum length (in terms of number of frames) of the features
# in the examples.
#
# To make sense of the egs script, it may be necessary to put an "exit 1"
# command immediately after stage 3.  Then, inspect
# exp/<your-dir>/egs/temp/ranges.* . The ranges files specify the examples that
# will be created, and which archives they will be stored in.  Each line of
# ranges.* has the following form:
#    <utt-id> <local-ark-indx> <global-ark-indx> <start-frame> <end-frame> <spk-id>
# For example:
#    100304-f-sre2006-kacg-A 1 2 4079 881 23

# If you're satisfied with the number of archives (e.g., 50-150 archives is
# reasonable) and with the number of examples per speaker (e.g., 1000-5000
# is reasonable) then you can let the script continue to the later stages.
# Otherwise, try increasing or decreasing the --num-repeats option.  You might
# need to fiddle with --frames-per-iter.  Increasing this value decreases the
# the number of archives and increases the number of examples per archive.
# Decreasing this value increases the number of archives, while decreasing the
# number of examples per archive.:
if [ $stage -le 12 ]; then
  echo "$0: Getting neural network training egs";
  # dump egs.
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $egs_dir/storage ]; then
    utils/create_split_dir.pl \
     /export/b{03,04,05,06}/$USER/kaldi-data/egs/librispeech/s5b/xvector-$(date +'%m_%d_%H_%M')/$egs_dir/storage $egs_dir/storage
  fi
  sid/nnet3/xvector/get_egs.sh --cmd "$train_cmd" \
    --nj 8 \
    --stage 0 \
    --frames-per-iter 1000000000 \
    --frames-per-iter-diagnostic 100000 \
    --min-frames-per-chunk 200 \
    --max-frames-per-chunk 400 \
    --num-diagnostic-archives 3 \
    --num-repeats 50 \
    "$data" $egs_dir
fi

${cuda_cmd} --gpu ${ngpu} $nnet_dir/log/train.log \
CUDA_VISIBLE_DEVICES=`free-gpu -n 1`  python scripts/train.py --data_dir $data --train_num_egs $train_num_egs --valid_num_egs $valid_num_egs \
	--clean_data_dir $clean_data --epochs $epochs --batch_size $batch_size --loss $loss --metric $metric --optimizer $optimizer --lr $lr \
	--arch $arch --feat_dim $feat_dim --num_classes $num_classes --multi_gpu $ngpu --seed $seed --num_workers $num_workers \
	--use_relu $use_relu --train_egs_dir $egs_dir --valid_egs_dir $egs_dir --use_tfb --use_multi $nnet_dir 
