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

void ComputeAndSubtractMean(
    std::map<std::string, Vector<BaseFloat> *> utt2ivector,
    Vector<BaseFloat> *mean_out) {
  int32 dim = utt2ivector.begin()->second->Dim();
  size_t num_ivectors = utt2ivector.size();
  Vector<double> mean(dim);
  std::map<std::string, Vector<BaseFloat> *>::iterator iter;
  for (iter = utt2ivector.begin(); iter != utt2ivector.end(); ++iter)
    mean.AddVec(1.0 / num_ivectors, *(iter->second));
  mean_out->Resize(dim);
  mean_out->CopyFromVec(mean);
  for (iter = utt2ivector.begin(); iter != utt2ivector.end(); ++iter)
    iter->second->AddVec(-1.0, *mean_out);
}

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

void NoisePrior::ComputeMeanAndCovariance(
    const MatrixBase<double> &noise_vectors,
    VectorBase<double> *mean,
    MatrixBase<double> *covariance) const {
  for (int32 i = 0; i < noise_vectors.NumRows(); i++) {
    mean.AddVec(1.0, noise_vectors(i));
  }
  mean.scale(1.0/noise_vectors.NumRows());
}

// "float" version of ComputeMeanAndCovariance
void NoisePrior::ComputeMeanAndCovariance(
    const MatrixBase<float> &noise_vectors,
    VectorBase<float> *mean,
    MatrixBase<float> *covariance) const {
  Matrix<double> tmp(noise_vectors), tmp_cov;
  Vector<double> tmp_mean;
  ComputeMeanAndCovariance(tmp, &tmp_mean, &tmp_cov);
  mean->CopyFromVec(tmp_mean);
  covariance->CopyFromMat(tmp_cov);
}

void EstimatePriorParameters(
    VectorBase<double> &mean,
    MatrixBase<double> &covariance) const {
}

// "float" version of EstimatePriorParameters
void EstimatePriorParameters(
    VectorBase<float> &mean,
    MatrixBase<float> &covariance) const {
  Vector<double> tmp_mean(mean);
  Matrix<double> tmp_cov(covariance);
  EstimatePriorParameters(tmp_mean, tmp_cov);
}


} // namespace kaldi
