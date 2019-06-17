import sys, os

db_dir = sys.argv[1]
data_dir = sys.argv[2]

for gender in ["f", "m"]:
  data_dir_enroll = data_dir + "/reddots_" + gender + "_enroll"
  data_dir_test = data_dir + "/reddots_" + gender + "_test"

  if not os.path.exists(data_dir_enroll):
    os.makedirs(data_dir_enroll)
  if not os.path.exists(data_dir_test):
    os.makedirs(data_dir_test)
  # Handle enroll
  models_filename = db_dir + "/ndx/" + gender + "_part_01.trn"
  models_fi = open(models_filename, 'r').readlines()
  utt2spk_enroll_fi = open(data_dir_enroll + "/utt2spk", 'w')
  wav_enroll_fi = open(data_dir_enroll + "/wav.scp", 'w')

  # Handle trials
  trials_filename = db_dir + "/ndx/" + gender + "_part_01.ndx"
  trials_in_fi = open(trials_filename, 'r').readlines()
  trials_out_fi = open(data_dir_test + "/trials", 'w')

  # Handle test
  utt2spk_test_fi = open(data_dir_test + "/utt2spk", 'w')
  wav_test_fi = open(data_dir_test + "/wav.scp", 'w')

  for line in models_fi:
    spk_id, utts = line.split(' ')
    utts = utts.split(',')
    for utt in utts:
      utt_parts = utt.split('/')[1].split('_')
      utt_id = spk_id + "_" + utt_parts[0]
      utt2spk_enroll_fi.write(utt_id + " " + spk_id + "\n")
      wav_path = db_dir + "/pcm/" + utt.rstrip() + ".wav"
      wav_enroll_fi.write(utt_id + " " + wav_path + "\n")

  for line in trials_in_fi:
    toks = line.split(',')
    spk_id = toks[0]
    utt = toks[1]
    utt_parts = utt.split('/')[1].split('_')
    utt_id = utt_parts[1] + "_" + utt_parts[2]  + "_" + utt_parts[0]
    is_target_correct = toks[2]
    is_target_wrong = toks[3]
    is_imposter_correct = toks[4]
    is_imposter_wrong = toks[5]

    if is_target_correct == 'Y':
      trials_out_fi.write(spk_id + " " + utt_id + " " + "target\n")
    if is_imposter_correct == 'Y':
      trials_out_fi.write(spk_id + " " + utt_id + " " + "nontarget\n")

    utt2spk_test_fi.write(utt_id + " " + utt_id + "\n")
    wav_path = db_dir + "/pcm/" + utt + ".wav"
    wav_test_fi.write(utt_id + " " + wav_path.rstrip() + "\n")

