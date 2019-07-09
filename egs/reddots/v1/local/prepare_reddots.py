#!/usr/bin/env python3
import sys, os

db_dir = sys.argv[1]
data_dir = sys.argv[2]

for gender in ["f", "m"]:
  data_dir_enroll = data_dir + "/reddots_enroll"
  data_dir_test = data_dir + "/reddots_test"

  # Handle enroll
  for part in ["part_01","part_02","part_03"]:
    models_filename = db_dir + "/ndx/" + gender + "_" + part + ".trn"
    models_fi = open(models_filename, 'r').readlines()
    enroll_dir = data_dir_enroll + "_" + gender + "_" + part
    if not os.path.exists(enroll_dir):
      os.makedirs(enroll_dir)
    utt2spk_enroll_fi = open(enroll_dir  + "/utt2spk", 'w')
    wav_enroll_fi = open(enroll_dir + "/wav.scp", 'w')
    
    for line in models_fi:
      spk_sent, utts = line.split(' ')
      spk_sent = spk_sent.replace('_','')
      utts = utts.split(',')
      for utt in utts:
        utt_parts = utt.split('/')[1].split('_')
        utt_id = spk_sent + "-" + utt_parts[0]
        utt2spk_enroll_fi.write(utt_id + " " + spk_sent + "\n")
        wav_path = db_dir + "/pcm/" + utt.rstrip() + ".wav"
        wav_enroll_fi.write(utt_id + " " + wav_path + "\n")


  # Handle trials
    trials_filename = db_dir + "/ndx/" + gender + "_" + part + ".ndx"
    trials_in_fi = open(trials_filename, 'r').readlines()
    test_dir = data_dir_test + "_" + gender + "_" + part
    if not os.path.exists(test_dir):
      os.makedirs(test_dir)
    trials_out_fi_tar_cor = open(test_dir + "/trials_tar_cor", 'w')
    trials_out_fi_tar_wr = open(test_dir + "/trials_tar_wr", 'w')
    trials_out_fi_imp_cor = open(test_dir + "/trials_imp_cor", 'w')
    trials_out_fi_imp_wr = open(test_dir + "/trials_imp_wr", 'w')


  # Handle test
    utt2spk_test_fi = open(test_dir + "/utt2spk", 'w')
    wav_test_fi = open(test_dir + "/wav.scp", 'w')

    for line in trials_in_fi:
      toks = line.strip('\n').split(',')
      spk_sent = toks[0]
      spk_sent = spk_sent.replace('_','')
      utt = toks[1]
      utt_parts = utt.split('/')[1].split('_')
      utt_id = spk_sent + "-" + utt_parts[0]
      is_target_correct = toks[2]
      is_target_wrong = toks[3]
      is_imposter_correct = toks[4]
      is_imposter_wrong = toks[5]

      if is_target_correct == 'Y':
        trials_out_fi_tar_cor.write(spk_sent + " " + utt_id + " " + "target\n")
      elif is_target_wrong == 'Y':
        trials_out_fi_tar_wr.write(spk_sent + " " + utt_id + " " + "nontarget\n")
      elif is_imposter_correct == 'Y':
        trials_out_fi_imp_cor.write(spk_sent + " " + utt_id + " " + "nontarget\n")
      elif is_imposter_wrong == 'Y':
        trials_out_fi_imp_wr.write(spk_sent + " " + utt_id + " " + "nontarget\n")

      utt2spk_test_fi.write(utt_id + " " + utt_id + "\n")
      wav_path = db_dir + "/pcm/" + utt + ".wav"
      wav_test_fi.write(utt_id + " " + wav_path.rstrip() + "\n")

