import sys

def split_egs(egs_dir, filename):
    with open("{}/{}".format(egs_dir, filename), 'r') as fh:
        content = fh.readlines()
    assert len(content) == 179865 
    num_split = 5
    num_utt_per_file = int(len(content) / num_split)
    for i in range(num_split):
        start_idx, end_idx = num_utt_per_file * i, num_utt_per_file * (i + 1)
        split_filename = "{}_{}.scp".format(filename[:-4], i)
        with open("{}/split/{}".format(egs_dir, split_filename), 'w') as fh:
            for j in range(start_idx, end_idx):
                fh.write(content[j])
    return 0

def main():
    egs_dir = "/export/b02/zili/sid/sre16_pytorch_v1/exp/xvector_pytorch/egs" 
    for i in range(1, 143):
        filename = "egs.{}.scp".format(i)
        split_egs(egs_dir, filename)
    return 0

if __name__ == "__main__":
    main()
