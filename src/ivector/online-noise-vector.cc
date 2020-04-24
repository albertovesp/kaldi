// ivector/online-noise-vector.cc

// Copyright 2020  Johns Hopkins University (author: Desh Raj)

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

#include "ivector/online-noise-vector.h"

namespace kaldi {

void OnlineNoisePrior::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<OnlineNoisePrior>");
  mu_n_.Write(os, binary);
  a_.Write(os, binary);
  B_.Write(os, binary);
  Lambda_n_.Write(os, binary);
  Lambda_s_.Write(os, binary);
  WriteToken(os, binary, "</OnlineNoisePrior>");
}

void OnlineNoisePrior::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<OnlineNoisePrior>");
  mu_n_.Read(is, binary);
  a_.Read(is, binary);
  B_.Read(is, binary);
  Lambda_n_.Read(is, binary);
  Lambda_s_.Read(is, binary);
  ExpectToken(is, binary, "</OnlineNoisePrior>");
}

int32 OnlineNoisePrior::Dim() const {
  return 2*a_.Dim();
}

void OnlineNoisePrior::EstimatePriorParameters(
    const VectorBase<BaseFloat> &mean,
    const SpMatrix<BaseFloat> &covariance,
    int32 dim) {
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


OnlineNoiseVector::OnlineNoiseVector(
    const OnlineNoisePrior &noise_prior,
    const int32 period):
    period_(period) {
  dim_ = noise_prior.Dim();
  current_vector_ = Vector<BaseFloat>(dim_);
  prior_.mu_n_ = noise_prior.mu_n_;
  prior_.a_ = noise_prior.a_;
  prior_.B_ = noise_prior.B_;
  prior_.Lambda_n_ = noise_prior.Lambda_n_;
  prior_.Lambda_s_ = noise_prior.Lambda_s_;
  prior_.r_s_ = noise_prior.r_s_;
  prior_.r_n_ = noise_prior.r_n_;
}

void OnlineNoiseVector::ExtractVectors(
    const Matrix<BaseFloat> &feats,
    const std::vector<bool> &silence_decisions,
    Matrix<BaseFloat> *noise_vectors) {
  int32 num_rows = (feats.NumRows() + period_ - 1)/period_;
  noise_vectors->Resize(num_rows, dim_);
  for (int32 i = 0; i < num_rows; ++i) {
    SubMatrix<BaseFloat> cur_feats(feats, i*period_, period_, 0, dim_/2);
    std::vector<bool>::const_iterator first = silence_decisions.begin() + i*period_;
    std::vector<bool>::const_iterator last = silence_decisions.begin() + (i+1)*period_;
    std::vector<bool> cur_decisions(first, last);
    UpdateVector(cur_feats, cur_decisions);
    UpdateScalingParams(cur_feats, cur_decisions);
    noise_vectors->CopyRowFromVec(current_vector_, i);
  }
}

void OnlineNoiseVector::ExtractVectors(
    const Matrix<BaseFloat> &feats,
    Matrix<BaseFloat> *noise_vectors) {
  int32 num_rows = (feats.NumRows() + period_ - 1)/period_;
  noise_vectors->Resize(num_rows, dim_);
  Vector<BaseFloat> noise_vec(dim_);
  SubVector<BaseFloat> sil_mean(noise_vec, 0, dim_/2);
  SubVector<BaseFloat> speech_mean(noise_vec, dim_/2, dim_/2);
  sil_mean.AddVec(1.0, prior_.mu_n_);
  speech_mean.AddVec(1.0, prior_.a_);
  speech_mean.AddMatVec(1.0, prior_.B_, kNoTrans, prior_.mu_n_, 1.0);
  for (int32 i = 0; i < num_rows; ++i) {
    noise_vectors->CopyRowFromVec(noise_vec, i);
  }
}

void OnlineNoiseVector::UpdateVector(
    SubMatrix<BaseFloat> &feats,
    std::vector<bool> &silence_decisions) {
  // We first compute the sufficient statistics for the new
  // chunk of data (i.e., for which we have silence decisions
  // in silence_frames. We need, for both speech and noise
  // frames, the number of frames, sum of all frames, and
  // the variance of all frames.
  int32 num_speech = 0, num_noise = 0, dim = dim_/2;
  Vector<BaseFloat> speech_sum(dim), noise_sum(dim);
  Matrix<BaseFloat> speech_var(dim, dim), noise_var(dim, dim);
  Vector<BaseFloat> cur_frame(dim);

  for (int32 i = 0; i < period_; ++i) {
    Vector<BaseFloat> cur_vec(dim);
    cur_vec.CopyFromVec(feats.Row(i));
    if (silence_decisions[i] == true) {
      // This is a silence frame
      num_noise++;
      noise_sum.AddVec(1.0, cur_vec);
      noise_var.AddVecVec(1.0, cur_vec, cur_vec);  
    } else {
      // This is a speech frame
      num_speech++;
      speech_sum.AddVec(1.0, cur_vec);
      speech_var.AddVecVec(1.0, cur_vec, cur_vec);
    }
  }

  // See paper for the math for this estimation method
  Matrix<BaseFloat> K(2*dim, 2*dim);
  Vector<BaseFloat> Q(2*dim);
  
  // Computing the matrix K
  SubMatrix<BaseFloat> K_11(K, 0, dim, 0, dim), K_12(K, 0, dim, dim, dim),
    K_21(K, dim, dim, 0, dim), K_22(K, dim, dim, dim, dim);
  K_11.AddMat(1.0, prior_.Lambda_s_);
  K_11.Scale(1.0 + prior_.r_s_*num_speech);
  K_12.AddMatMat(-1.0, prior_.Lambda_s_, kNoTrans, prior_.B_, kNoTrans, 0);
  K_21.AddMatMat(-1.0, prior_.B_, kTrans, prior_.Lambda_s_, kNoTrans, 0);
  K_22.AddMat(1.0, prior_.Lambda_n_);
  K_22.Scale(1 + prior_.r_n_*num_noise);
  {
    Matrix<BaseFloat> temp(dim, dim);
    temp.AddMatMat(1.0, prior_.B_, kTrans, prior_.Lambda_s_, kNoTrans, 0);
    K_22.AddMatMat(1.0, temp, kNoTrans, prior_.B_, kNoTrans, 1);
  }

  // Computing the vector Q
  SubVector<BaseFloat> Q_1(Q, 0, dim), Q_2(Q, dim, dim);
  {
    Vector<BaseFloat> temp = prior_.a_;
    temp.AddVec(prior_.r_s_, speech_sum);
    Q_1.AddMatVec(1.0, prior_.Lambda_s_, kNoTrans, temp, 0.0);
  }
  {
    Vector<BaseFloat> temp = prior_.mu_n_;
    temp.AddVec(prior_.r_n_, noise_sum);
    Q_2.AddMatVec(1.0, prior_.Lambda_n_, kNoTrans, temp, 0.0);
    temp.AddMatVec(0.0, prior_.Lambda_s_, kNoTrans, prior_.a_, 1.0);
    Q_2.AddMatVec(1.0, prior_.B_, kTrans, temp, 1.0);
  }

  // Compute the nvector from K and Q
  K.Invert();
  current_vector_.AddMatVec(1.0, K, kNoTrans, Q, 0.0);
}

void OnlineNoiseVector::UpdateScalingParams(
    SubMatrix<BaseFloat> &feats,
    std::vector<bool> &silence_decisions) {
  int32 num_speech = 0, num_noise = 0, dim = dim_/2;
  Matrix<BaseFloat> speech_var(dim, dim), noise_var(dim, dim);
  
  Vector<BaseFloat> cur_frame(dim);
  SubVector<BaseFloat> noise_vec(current_vector_, 0, dim);
  SubVector<BaseFloat> speech_vec(current_vector_, dim, dim);
  for (int32 i = 0; i < period_; ++i) {
    cur_frame.CopyFromVec(feats.Row(i));
    if (silence_decisions[i] == true) {
      // This is a silence frame
      num_noise++;
      cur_frame.AddVec(-1.0, noise_vec);
      noise_var.AddVecVec(1.0, cur_frame, cur_frame);  
    } else {
      // This is a speech frame
      num_speech++;
      cur_frame.AddVec(-1.0, speech_vec);
      speech_var.AddVecVec(1.0, cur_frame, cur_frame);
    }
  }
  
  if (num_speech > 0) { 
    prior_.r_s_ = (dim * num_speech) / 
      TraceMatMat(prior_.Lambda_s_, speech_var);
  }
  if (num_noise > 0) {
  prior_.r_n_ = (dim * num_noise) / 
    TraceMatMat(prior_.Lambda_n_, noise_var);
  }
}

OnlineNoiseVector::~OnlineNoiseVector() {
  // Delete objects owned here.
}

}  // namespace kaldi
