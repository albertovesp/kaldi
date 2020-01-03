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
  Matrix<BaseFloat> covariance_(covariance);
  SubVector<BaseFloat> mu_n(mean, 0, dim/2);
  SubVector<BaseFloat> mu_s(mean, dim/2, dim);
  SubMatrix<BaseFloat> cov_nn(covariance_, 0, dim/2, 0, dim/2); 
  SubMatrix<BaseFloat> cov_sn(covariance_, dim/2, dim, 0, dim/2); 
  SubMatrix<BaseFloat> cov_ss(covariance_, dim/2, dim, dim/2, dim);
  Matrix<BaseFloat> Lambda_nn(cov_nn), Lambda_sn(cov_sn), 
    Lambda_ss(cov_ss), cov_ss_(cov_ss);
  Lambda_nn.Invert(); 
  Lambda_sn.Invert(); 
  Lambda_ss.Invert(); 
  mu_n_ = mu_n;
  Lambda_n_ = Lambda_nn;
  Lambda_s_ = Lambda_ss;
  // B = - (Lambda_ss)^-1 Lambda_sn
  B_.AddMatMat(-1.0, cov_ss_, kNoTrans, Lambda_sn, kNoTrans, 0);
  // a = mu_s - B mu_n
  a_.CopyFromVec(mu_s);
  a_.AddMatVec(-1.0, B_, kNoTrans, mu_n_, 1);
}


} // namespace kaldi
