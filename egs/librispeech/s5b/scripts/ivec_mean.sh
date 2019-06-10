#!/bin/bash
dataset=sre_combined

ivector-mean ark:data/$dataset/spk2utt scp:exp/xvectors_${dataset}/xvector.scp ark,scp:exp/xvectors_${dataset}/spk_xvector.ark,exp/xvectors_${dataset}/spk_xvector.scp ark,t:exp/xvectors_${dataset}/num_utts.ark || exit 1;
