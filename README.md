[![Build Status](https://travis-ci.com/kaldi-asr/kaldi.svg?branch=master)](https://travis-ci.com/kaldi-asr/kaldi)
<<<<<<< HEAD
Desh's fork of Kaldi
=======
[![Gitpod Ready-to-Code](https://img.shields.io/badge/Gitpod-Ready--to--Code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/kaldi-asr/kaldi) 
Kaldi Speech Recognition Toolkit
>>>>>>> f71d7ba0a9b54b5ee955c44dbe4f935baa603ea2
================================

For installation and other information, see the [Kaldi-trunk](https://github.com/kaldi-asr/kaldi)

Branches in this fork of Kaldi contain code for different papers, as listed below. 

   1. [**Probing the information encoded in x-vectors**](https://arxiv.org/pdf/1909.06351.pdf), Desh Raj, David Snyder, Daniel Povey, and Sanjeev Khudanpur. IEEE ASRU 2019: Check out the [reddots](https://github.com/desh2608/kaldi/tree/reddots) branch and look under egs/reddots/v1.
   
   2. [**Multi-class spectral clustering with overlaps for speaker diarization**](https://arxiv.org/pdf/2011.02900.pdf). Desh Raj, Zili Huang, and Sanjeev Khudanpur. IEEE SLT 2021. Check out the [slt21_spectral](https://github.com/desh2608/kaldi/tree/slt21_spectral) branch and look under the files changed in [this commit](https://github.com/desh2608/kaldi/commit/2e07a9eae732b334e33d8b9d1b4f3a2c251b4b2c). The actual overlap-aware SC code can be found in [my fork](https://github.com/desh2608/scikit-learn/blob/fe74ed9573160d87aa5e40d8bd9d9af20283b3bc/sklearn/cluster/_spectral.py#L23) of scikit-learn. Detailed description of the code is available [here](https://desh2608.github.io/pages/overlap-aware-sc/).
   
## Citation

If you use any of the above code, please cite the respective papers.

## Contact

For any questions, please contact me at draj@cs.jhu.edu or r.desh26@gmail.com.
