# This script is called from run.sh and it prepares
# a subset of the RIR augmentations and combines them
# with the utterances to generate the final train,
# dev, and test sets.

stage=3
rir_dir=data/rir
keep_frac=0.1
num_train_replicas=6
rooms="mediumroom smallroom"

. utils/parse_options.sh

if [ $stage -le 3 ]; then
  # Combine large, medium, and small room RIR lists
  # and select a subset of them according to 
  # given parameter.
  if [ -d "$rir_dir" ]; then rm -Rf $rir_dir; fi
  mkdir -p $rir_dir
  touch $rir_dir/rir_list
  touch $rir_dir/room_info
   
  for room in $rooms; do
    cat RIRS_NOISES/simulated_rirs/$room/rir_list >> $rir_dir/rir_list
    cat RIRS_NOISES/simulated_rirs/$room/room_info >> $rir_dir/room_info
  done
  
  # To select a subset, print every nth line from rir_list
  # into a new file, where n=1/keep_frac
  NUM=`echo print 1/$keep_frac | python`
  awk -v NUM=$NUM 'NR % NUM == 0' $rir_dir/rir_list > $rir_dir/rir_list_subset
fi

if [ $stage -le 4 ]; then
  # Create reverberated copies of the data
  steps/data/reverberate_data_dir.py \
    --rir-set-parameters "$rir_dir/rir_list_subset" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications $num_train_replicas \
    --source-sampling-rate 16000 \
    --store-rir-ids true \
    data/train_clean_100 data/train_rir_100

  steps/data/reverberate_data_dir.py \
    --rir-set-parameters "$rir_dir/rir_list_subset" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    --store-rir-ids true \
    data/dev_clean data/dev_rir

  steps/data/reverberate_data_dir.py \
    --rir-set-parameters "$rir_dir/rir_list_subset" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    --store-rir-ids true \
    data/test_clean data/test_rir  
fi

if [ $stage -le 5 ]; then
  for datadir in train_rir_100 dev_rir test_rir; do
    cut -d' ' -f2 data/${datadir}/utt2spk | \
      cut -d'-' -f 1,2 - > data/${datadir}/rooms.temp
    join -t ' ' data/${datadir}/rooms.temp data/rir/room_info -a1 | \
      cut -d' ' -f2- > data/${datadir}/params.temp
    cut -d' ' -f1 data/${datadir}/utt2spk | paste -d' ' - data/${datadir}/params.temp \
      > data/${datadir}/utt2params
    rm data/${datadir}/rooms.temp
    rm data/${datadir}/params.temp
  done
fi
