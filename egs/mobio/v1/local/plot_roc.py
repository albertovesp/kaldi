#!/usr/bin/env python

# Copyright 2018 Johns Hopkins University (author: Yiming Wang)
# Apache 2.0

""" This script prepares the speech commands data into kaldi format.
"""


import argparse
import os
import io
import sys
import re
import numpy as np

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        """This script requires matplotlib.
        Please install it to generate plots.
        If you are on a cluster where you do not have admin rights you could
        try using virtualenv.""")
    
def PlotRoc(fnrs_list, fprs_list, color_val_list, name_list, savedir):
    assert len(fnrs_list) == len(fprs_list) and \
        len(fnrs_list) == len(color_val_list) and len(fnrs_list) == len(name_list)
    fig = plt.figure()
    roc_plots = []
    for i in range(len(fnrs_list)):
        fnrs = fnrs_list[i]
        fprs = fprs_list[i]
        color_val = color_val_list[i]
        name = name_list[i]
        roc_plot_handle, = plt.plot([fpr * 100 for fpr in fprs],
            [fnr * 100 for fnr in fnrs], color=color_val,
            linestyle="--", label="{}".format(name)
        )
        roc_plots.append(roc_plot_handle)

    plt.xlabel('False Alarms (%)')
    plt.ylabel('False Rejects (%)')
    plt.xlim((0, 20))
    plt.ylim((0, 20))
    lgd = plt.legend(handles=roc_plots, loc='lower center',
        bbox_to_anchor=(0.5, -0.2 + len(fnrs_list) * -0.1),
        ncol=1, borderaxespad=0.)
    plt.grid(True)
    fig.suptitle("ROC curve")
    figfile_name = os.path.join(savedir, 'roc.pdf')
    plt.savefig(figfile_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    print("Saved ROC curves as " + figfile_name)

def PlotRoc2(fnrs_list, fphs_list, color_val_list, name_list, savedir):
    assert len(fnrs_list) == len(fphs_list) and \
        len(fnrs_list) == len(color_val_list) and len(fnrs_list) == len(name_list)
    fig = plt.figure()
    roc_plots = []
    for i in range(len(fnrs_list)):
        fnrs = fnrs_list[i]
        fphs = fphs_list[i]
        color_val = color_val_list[i]
        name = name_list[i]
        roc_plot_handle, = plt.plot(fphs,
            [fnr * 100 for fnr in fnrs], color=color_val,
            linestyle="--", label="{}".format(name)
        )
        roc_plots.append(roc_plot_handle)

    plt.xlabel('False Alarm per Hour')
    plt.ylabel('False Rejects (%)')
    #plt.xlim((0, 5))
    plt.xticks(np.arange(0, 5, step=0.5))
    #plt.ylim((0, 20))
    plt.yticks(np.arange(0, 10, step=0.1))
    lgd = plt.legend(handles=roc_plots, loc='lower center',
        bbox_to_anchor=(0.5, -0.2 + len(fnrs_list) * -0.1),
        ncol=1, borderaxespad=0.)
    plt.grid(True)
    fig.suptitle("ROC curve")
    figfile_name = os.path.join(savedir, 'roc2.pdf')
    plt.savefig(figfile_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    print("Saved ROC2 curves as " + figfile_name)


def main():
    parser = argparse.ArgumentParser(description="""Computes metrics for evalutuon.""")
    parser.add_argument('comparison_path', type=str, nargs='+',
                        help='paths to result file')

    args = parser.parse_args()
    if (args.comparison_path is not None and len(args.comparison_path) > 6):
        raise Exception(
            """max 6 comparison paths can be specified.
            If you want to compare with more comparison_path, you would have to
            carefully tune the plot_colors variable which specified colors used
            for plotting.""")

    g_plot_colors = ['red', 'blue', 'green', 'black', 'magenta', 'yellow', 'cyan']

    pattern = r"precision: (\d+\.\d*)  recall: (\d+\.\d*)  FPR: (\d+\.\d*)  FNR: (\d+\.\d*)  FP per hour: (\d+\.\d*)  total: \d+"
    prog = re.compile(pattern)
    
    fig = plt.figure()
    roc_plots = []
    fnrs_list1, fnrs_list2, fprs_list, fph_list, color_val_list, name_list = [], [], [], [], [], []
    for index, path in enumerate(args.comparison_path):
        if index == 0:
            savedir = os.path.dirname(path)

        with open(path, 'r') as f:
            lines = f.readlines()

        precision = []
        recall = []
        FPR = []
        FNR = []
        FP_per_hour = []
        for line in lines:
            m = prog.match(line)
            if m:
                precision.append(float(m.group(1)))
                recall.append(float(m.group(2)))
                FPR.append(float(m.group(3)))
                FNR.append(float(m.group(4)))
                FP_per_hour.append(float(m.group(5)))

        name_list.append(os.path.dirname(path))
        
        fprs, fnrs = tuple(zip(*sorted(zip(FPR, FNR), key=lambda x: (x[0], -x[1]))))
        fprs_list.append(fprs)
        fnrs_list1.append(fnrs)

        fph, fnrs = tuple(zip(*sorted(zip(FP_per_hour, FNR), key=lambda x: (x[0], -x[1]))))
        fph_list.append(fph)
        fnrs_list2.append(fnrs)
    
    color_val_list = g_plot_colors[:len(args.comparison_path)]
    PlotRoc(fnrs_list1, fprs_list, color_val_list, name_list, savedir)
    PlotRoc2(fnrs_list2, fph_list, color_val_list, name_list, savedir)

if __name__ == "__main__":
    main()
