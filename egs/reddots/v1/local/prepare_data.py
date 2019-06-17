#!/usr/bin/env python3
# Copyright 2019  Desh Raj (Johns Hopkins University)

import sys
import os
import subprocess
import io

def get_spk_stats(db_dir, data_dir):
  stats_fh = open(data_dir + '/spk_stats', 'w')
  with open(db_dir + '/infos/stat_reddots_r2015q4_v1.txt', 'r') as f:
    next(f)
    for line in f:
      spkid, num_utts, _ = line.split(' ')
      stats_fh.write(spkid + ' ' + num_utts + '\n')
  stats_fh.close()

def prepare_utt2spk_wav(db_dir, data_dir, gender):
  utt2spk_fh = open(data_dir + '/utt2spk', 'a+')
  wav_fh = open(data_dir + '/wav.scp', 'a+')

  for file in ["01", "02", "03"]:
    models_filename = db_dir + "/ndx/" + gender + "_part_" + file + ".trn"
    models_fi = open(models_filename, 'r').readlines()
    for line in models_fi:
      spk_id, utts = line.split(' ')
      utts = utts.split(',')
      for utt in utts:
        utt_parts = utt.split('/')[1].split('_')
        utt_id = spk_id + "-" + utt_parts[0]
        utt2spk_fh.write(utt_id + " " + spk_id + "\n")
        wav_path = db_dir + "/pcm/" + utt.rstrip() + ".wav"
        wav_fh.write(utt_id + " " + wav_path + "\n")
    

def prepare_utt2text(db_dir, data_dir):
  script_fh = open(db_dir + '/infos/script.txt', 'r', encoding='latin-1')
  text_fh = open(data_dir + '/text', 'w', encoding='latin-1')
  for line in script_fh.readlines():
    utt, text = line.split(';')
    utt_parts = utt.split('_')
    spk_id = '_'.join(utt_parts[1:])
    utt_id = spk_id + '-' + utt_parts[0]
    text_fh.write(utt_id + ' ' + text)
  text_fh.close()
  command = 'sort -k1 -o '
  command += data_dir + '/text ' + data_dir + '/text'
  subprocess.call(command, shell=True)

def prepare_utt2gender(data_dir):
  utt2gender_fh = open(data_dir + '/utt2gender', 'w')
  utt2spk_fh = open(data_dir + '/utt2spk' , 'r')
  for line in utt2spk_fh.readlines():
    utt_id, spk_id = line.split(' ')
    gender = spk_id[0]
    utt2gender_fh.write(utt_id + ' ' + gender + '\n')
  utt2gender_fh.close()
  utt2spk_fh.close()

def prepare_utt2dur(data_dir):
  command = 'utils/data/get_utt2dur.sh '
  command += data_dir
  subprocess.call(command, shell=True)

if __name__=="__main__":
  db_dir = sys.argv[1]
  data_dir = sys.argv[2]
  get_spk_stats(db_dir, data_dir)
  for gender in ['m', 'f']:
    prepare_utt2spk_wav(db_dir, data_dir, gender)
  prepare_utt2text(db_dir, data_dir)
  prepare_utt2gender(data_dir)
  prepare_utt2dur(data_dir)
