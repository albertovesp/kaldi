// ivectorbin/ivector-speaker-matrix.cc

// Copyright 2019  Desh Raj

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/stl-utils.h"
#include "util/kaldi-thread.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
      "Copy speaker-level i-vectors into a matrix. Note that the input is\n"
      "online i-vectors, i.e., i-vectors copied across frames.\n"
      "Usage: ivector-speaker-matrix [options] <spk2utt> \n"
      "<ivectors-rspecifier> <ivectors-matrix-out>\n"
      "e.g.: \n"
      "  ivector-speaker-matrix spk2utt scp:ivector_online.scp ivector.mat\n";

    ParseOptions po(usage); 
    
    bool binary = true;
    bool bias = false;
    
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("bias", &bias, "Add a bias column of zeros to matrix.");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string spk2utt_rspecifier = po.GetArg(1),
      ivector_rspecifier = po.GetArg(2),
      ivector_mat_wxfilename = po.GetArg(3);

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    std::vector<Vector<BaseFloat> > ivectors;

    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();

      std::vector<std::string> &uttlist = spk2utt_reader.Value();
      std::string utt = uttlist[0];


      if (!ivector_reader.HasKey(utt)) {
        KALDI_ERR << "No iVector present in input for speaker " << spk;
      }

      Vector<BaseFloat> ivector = ivector_reader.Value(utt);
      ivectors.push_back(ivector);
    } 
    
    Matrix<BaseFloat> ivector_mat(ivectors.size() + int(bias), ivectors[0].Dim());

    for (size_t i = 0; i < ivectors.size(); i++) {
      ivector_mat.Row(i).CopyFromVec(ivectors[i]);
    }
    
    if (bias) {
      Vector<BaseFloat> zero_vec(ivectors[0].Dim());
      ivector_mat.Row(ivectors.size()).CopyFromVec(zero_vec);
    }

    ivector_mat.Transpose();

    WriteKaldiObject(ivector_mat, ivector_mat_wxfilename, binary);
  
    return 0;

  } catch(const std::exception &e) {
      std::cerr << e.what();
      return -1;
  }
}

