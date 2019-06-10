import kaldi_io

ark_info = "exp/xvector_pytorch/egs/egs.1.ark:43" 
s = kaldi_io.read_mat(ark_info)
ark_info_1 = "ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:{} ark:- |".format(ark_info) 
t = kaldi_io.read_mat(ark_info_1)
print("t", t)
