#!/usr/bin/env python3
# Copyright 2019 Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

import argparse, sys, random

def get_args():
  parser = argparse.ArgumentParser(description="Optionally add silence annotations"
                    " to annotations file, to simulate cases where test conditions"
                    " have silences", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--silence-ratio", type=float, dest="silence_ratio", default=0.1,
                      help="Fraction of total annotations to add as silence")

  parser.add_argument('--random-seed', type=int, dest="random_seed", default=0, 
                      help='seed to be used in the randomization of segment selection')

  parser.add_argument("in_file",
                      help="Annotations file")

  parser.add_argument("out_file",
                      help="File in which to write the output annotations with silence")

  print(' '.join(sys.argv))
  args = parser.parse_args()
  args = check_args(args)
  return args

def check_args(args):
  if args.silence_ratio < 0:
    raise Exception("--silence-ratio must be non-negative")
  return args

def read_annotations(file):
  with open(file, 'r') as fh:
    samples = fh.readlines()
  annotations = {}
  for sample in samples:
    meeting, channel, speaker, stime, etime, text = sample.strip().split(' ', 5)
    value = (channel, speaker, float(stime), float(etime), text)
    if meeting in annotations:
      annotations[meeting].append(value)
    else:
      annotations[meeting] = [value]
  return (annotations)

def compute_missing_segs(annotations):
  missing_segs = {}
  for meeting in annotations:
    value = annotations[meeting]
    sorted_values = sorted(value, key=lambda tup: tup[2])
    missing_values = []
    for i, tup_cur in enumerate(sorted_values[:-1]):
      tup_next = sorted_values[i+1]
      if (tup_cur[3] < tup_next[2]):
        missing_values.append((tup_cur[0], tup_cur[1], tup_cur[3], tup_next[2], 'sil'))
    missing_segs[meeting] = missing_values
  return missing_segs

def get_list_from_dict(segs_dict):
  segs_list = []
  for key in segs_dict:
    vals = segs_dict[key]
    for val in vals:
      val = [str(x) for x in val]
      segs_list.append([key] + val)
  return segs_list

def create_annotations_with_sil(voiced_segs, missing_segs, num_silence):
  voiced_segs_list = get_list_from_dict(voiced_segs)
  missing_segs_list = get_list_from_dict(missing_segs)
  missing_segs_selected = random.sample(missing_segs_list, num_silence)
  annotations = voiced_segs_list + missing_segs_selected
  annotations_sorted = sorted(annotations, key=lambda tup: tup[0])
  return annotations_sorted

def write_to_file(out_file, annotations):
  with open(out_file, 'w') as fh:
    for sample in annotations:
      fh.write(' '.join(sample))
      fh.write('\n')
  return

def main():
  args = get_args()
  random.seed(args.random_seed)

  annotations = read_annotations(args.in_file)
  num_total = sum([len(annotations[x]) for x in annotations])
  print("Number of annotations is {0}".format(num_total))
  missing_segs = compute_missing_segs(annotations)
  num_missing = sum([len(missing_segs[x]) for x in missing_segs])
  print("Number of missing segments is {0}".format(num_missing))
  num_silence = min(num_missing, int(args.silence_ratio*num_total))
  print("Adding {0} silence annotations".format(num_silence))

  annotations_with_sil = create_annotations_with_sil(annotations, missing_segs, num_silence)
  write_to_file(args.out_file, annotations_with_sil)

if __name__=="__main__":
  main()
