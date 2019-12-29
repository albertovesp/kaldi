// ivector/online-noise-vector.h

// Copyright 2019    Johns Hopkins University (Author: Desh Raj)


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

#ifndef KALDI_IVECTOR_ONLINE_NOISE_VECTOR_H_
#define KALDI_IVECTOR_ONLINE_NOISE_VECTOR_H_

#include <vector>
#include <algorithm>
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "gmm/model-common.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "itf/options-itf.h"
#include "util/common-utils.h"

namespace kaldi {

/* This code implements a Bayesian model for online estimation of speech
 * and noise vectors. First, we estimate tthe prior parameters from the 
 * training data:
 * pi = (mu_n, a, B, Lambda_n, Lambda_s),
 * where mu_n = mean of noise frames in training set,
 * a = mu_s + {Lambda_ss}^-1 Lambda_sn mu_n,
 * B = - {Lambda_ss}^-1 Lambda_sn,
 * Lambda's are the inverse of the covariance mattrices (i.e., they are the
 * precision matrices). 
 * For derivation, see related paper.
 *
 * After estimating the prior parameters, at inference time, the posteriors
 * are computed through an E-M like procedure.
*/

class NoisePrior {
 public:
  NoisePrior() { }

  explicit NoisePrior(const NoisePrior &other):
    mu_n_(other.mu_n_),
    a_(other.a_),
    B_(other.B_),
    Lambda_n_(other.Lambda_n_),
    Lambda_s_(other.Lambda_s_) {
  };

  /// Takes the mean and covariance matrix computed from the
  /// training data and estimates the prior parameters.
  void EstimatePriorParameters(const VectorBase<double> &mean,
                               const MatrixBase<double> &covariance) const;

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

 protected:
  Vector<double> mu_n_;  // mean of noise vectors.
  double a_;  // shift factor for mean of speech vectors.
  double B_;  // scale factor for mean of speech vectors.
  Matrix<double> Lambda_n_; // precision matrix for noise.
  Matrix<double> Lambda_s_; // precision matrix for speech.

 private:
  NoisePrior &operator = (const NoisePrior &other);  // disallow assignment


};

}  // namespace kaldi

#endif
