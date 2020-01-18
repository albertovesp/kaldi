// ivector/online-noise-vector.cc

// Copyright 2019     Johns Hopkins University (Author: Desh Raj)

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

#include <vector>
#include "ivector/online-noise-vector.h"

namespace kaldi {

void NoisePrior::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<NoisePrior>");
  mu_n_.Write(os, binary);
  a_.Write(os, binary);
  B_.Write(os, binary);
  Lambda_n_.Write(os, binary);
  Lambda_s_.Write(os, binary);
  WriteToken(os, binary, "</NoisePrior>");
}

void NoisePrior::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<NoisePrior>");
  mu_n_.Read(is, binary);
  a_.Read(is, binary);
  B_.Read(is, binary);
  Lambda_n_.Read(is, binary);
  Lambda_s_.Read(is, binary);
  ExpectToken(is, binary, "</NoisePrior>");
}

void NoisePrior::EstimatePriorParameters(
    const VectorBase<BaseFloat> &mean,
    const SpMatrix<BaseFloat> &covariance,
    const int32 dim) {
  SubVector<BaseFloat> mu_n(mean, 0, dim/2);
  SubVector<BaseFloat> mu_s(mean, dim/2, dim/2);
  Matrix<BaseFloat> Lambda(covariance);
  Lambda.Invert();
  SubMatrix<BaseFloat> Lambda_nn(Lambda, 0, dim/2, 0, dim/2); 
  SubMatrix<BaseFloat> Lambda_sn(Lambda, dim/2, dim/2, 0, dim/2); 
  SubMatrix<BaseFloat> Lambda_ss(Lambda, dim/2, dim/2, dim/2, dim/2);
  mu_n_ = mu_n;
  Lambda_n_ = Lambda_nn;
  Lambda_s_ = Lambda_ss;
  Matrix<BaseFloat> Lambda_sn_(Lambda_sn), Lambda_ss_inv(Lambda_ss);
  Lambda_ss_inv.Invert();
  // B = - (Lambda_ss_inv)^-1 Lambda_sn
  Matrix<BaseFloat> temp(dim/2, dim/2);
  temp.AddMatMat(-1.0, Lambda_ss_inv, kNoTrans, Lambda_sn_, kNoTrans, 0);
  B_ = temp;
  // a = mu_s - B mu_n
  a_ = mu_s;
  a_.AddMatVec(-1.0, temp, kNoTrans, mu_n_, 1);
}


} // namespace kaldi
