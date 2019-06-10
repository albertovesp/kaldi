#!/home/hzili1/tools/anaconda3/envs/py27/bin/python

import os
import sys

def create_segments_file(egs_dir, output_dir):
    file_list = os.listdir(egs_dir)
    for filename in file_list:
        if filename[:3] == "egs" and filename[-3:] == "scp":
            with open("{}/{}".format(egs_dir, filename), 'r') as fh:
                content = fh.readlines()
            f = open("{}/{}".format(output_dir, filename), 'w')
            for line in content:
                line = line.strip('\n')
                segment_name = line.split()[0]
                segment_name_split = segment_name.split('-')
                spk_id = segment_name_split[-1]
                duration = int(segment_name_split[-2])
                start_frame = int(segment_name_split[-3]) 
                uttname = "-".join(segment_name_split[:-3]) 
                f.write("{} {} {} {}\n".format(segment_name, uttname, start_frame, start_frame + duration))
            f.close()
        elif filename[:9] == "valid_egs" and filename[-3:] == "scp":
            with open("{}/{}".format(egs_dir, filename), 'r') as fh:
                content = fh.readlines()
            f = open("{}/{}".format(output_dir, filename), 'w')
            for line in content:
                line = line.strip('\n')
                segment_name = line.split()[0]
                segment_name_split = segment_name.split('-')
                spk_id = segment_name_split[-1]
                duration = int(segment_name_split[-2])
                start_frame = int(segment_name_split[-3]) 
                uttname = "-".join(segment_name_split[:-3]) 
                f.write("{} {} {} {}\n".format(segment_name, uttname, start_frame, start_frame + duration))
            f.close()
        else:
            continue
    return 0

def main():
    assert len(sys.argv) == 3
    egs_dir = sys.argv[1]
    output_dir = sys.argv[2]
    create_segments_file(egs_dir, output_dir)
    return 0

if __name__ == "__main__":
    main()
