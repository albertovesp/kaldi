import numpy as np
import kaldi_io
import sys
import os

def compute_mean_std(data_dir):
    data_list = []
    filelist = os.listdir(data_dir)
    for filename in filelist:
        if filename == "egs.1.ark" or filename == "egs.2.ark":
            print(filename)
            with open("{}/{}".format(data_dir, filename), 'rb') as fh:
                for key, mat in kaldi_io.read_mat_ark(fh):
                    data_list.append(mat)
    data_array = np.concatenate(data_list, axis=0)
    print(data_array.shape)
    print("np.max(data_array)", np.max(data_array))
    print("np.min(data_array)", np.min(data_array))
    mean = np.mean(data_array, axis=0)
    std = np.std(data_array, axis=0)
    print("mean", mean)
    print("std", std)
    np.save("{}/mean.npy".format(data_dir), mean)
    np.save("{}/std.npy".format(data_dir), std)
    return 0

def main():
    data_dir = sys.argv[1]
    compute_mean_std(data_dir)
    return 0

if __name__ == "__main__":
    main()
