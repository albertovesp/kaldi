// online2/online-nvector-feature.cc

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

#include "online2/online-nvector-feature.h"

namespace kaldi {

OnlineNvectorExtractionInfo::OnlineNvectorExtractionInfo(
    const OnlineNvectorExtractionConfig &config) {
  Init(config);
}

void OnlineNvectorExtractionInfo::Init(
    const OnlineNvectorExtractionConfig &config) {
  nvector_period = config.nvector_period;
  use_most_recent_nvector = config.use_most_recent_nvector;
  greedy_nvector_extractor = config.greedy_nvector_extractor;
  if (greedy_nvector_extractor && !use_most_recent_nvector) {
    KALDI_WARN << "--greedy-nvector-extractor=true implies "
               << "--use-most-recent-nvector=true";
    use_most_recent_nvector = true;
  }
  max_remembered_frames = config.max_remembered_frames;

  std::string note = "(note: this may be needed "
      "in the file supplied to --nvector-extractor-config)";
  if (config.noise_prior_rxfilename == "")
    KALDI_ERR << "--noise-prior option must be set " << note;
  ReadKaldiObject(config.noise_prior_rxfilename, &noise_prior);
  this->Check();
}

void OnlineNvectorExtractionInfo::Check() const {
  KALDI_ASSERT(nvector_period > 0);
  KALDI_ASSERT(max_remembered_frames >= 0);
}

// The class constructed in this way should never be used.
OnlineNvectorExtractionInfo::OnlineNvectorExtractionInfo():
    nvector_period(0), use_most_recent_nvector(true), 
    greedy_nvector_extractor(false), max_remembered_frames(0) { }


void OnlineNvectorEstimationParams::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<OnlineNvectorEstimationParams>");
  mu_n_.Write(os, binary);
  a_.Write(os, binary);
  B_.Write(os, binary);
  Lambda_n_.Write(os, binary);
  Lambda_s_.Write(os, binary);
  WriteToken(os, binary, "</OnlineNvectorEstimationParams>");
}

void OnlineNvectorEstimationParams::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<OnlineNvectorEstimationParams>");
  mu_n_.Read(is, binary);
  a_.Read(is, binary);
  B_.Read(is, binary);
  Lambda_n_.Read(is, binary);
  Lambda_s_.Read(is, binary);
  ExpectToken(is, binary, "</OnlineNvectorEstimationParams>");
}

void OnlineNvectorEstimationParams::EstimatePriorParameters(
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


OnlineNvectorFeature::OnlineNvectorFeature(
    const OnlineNvectorExtractionInfo &info,
    OnlineFeatureInterface *base_feature,
    OnlineNvectorEstimationStats *adaptation_state):
    info_(info),
    base_(base_feature),
    nvector_stats_(adaptation_state),
    num_frames_stats_(0) {
  info.Check();
  KALDI_ASSERT(base_feature != NULL);

}

int32 OnlineNvectorFeature::Dim() const {
  return info_.extractor.NvectorDim();
}

bool OnlineNvectorFeature::IsLastFrame(int32 frame) const {
  return base_->IsLastFrame(frame);
}

int32 OnlineNvectorFeature::NumFramesReady() const {
  KALDI_ASSERT(lda_ != NULL);
  return base_->NumFramesReady();
}

BaseFloat OnlineNvectorFeature::FrameShiftInSeconds() const {
  return base_->FrameShiftInSeconds();
}

void OnlineNvectorFeature::GetAdaptationState(
    OnlineNvectorEstimationStats *adaptation_state) const {
  adaptation_state = nvector_stats_;
}


void OnlineNvectorFeature::SetAdaptationState(
    const OnlineNvectorEstimationStats &adaptation_state) {
  nvector_stats_ = adaptation_state;
}

void OnlineNvectorFeature::UpdateNvector(
    const std::vector<std::pair<int32, bool> > &silence_frames) {
  // We first compute the sufficient statistics for the new
  // chunk of data (i.e., for which we have silence decisions
  // in silence_frames. We need, for both speech and noise
  // frames, the number of frames, sum of all frames, and
  // the variance of all frames.
  int32 num_speech = 0, num_noise = 0, dim = this->Dim();
  Vector<BaseFloat> speech_sum(dim), noise_sum(dim);
  Matrix<BaseFloat> speech_var(dim, dim), noise_var(dim, dim);
  Vector<BaseFloat> cur_frame;
  // Check that we have the frames we need to perform the computations
  KALDI_ASSERT(NumFramesReady() >= silence_frames.back()->first);

  std::vector<std::pair<int32, bool> >::iterator it;
  for (it = silence_frames.begin(); it != silence_frames.end(); it++) {
    base_->GetFrame(it->first, &cur_frame);
    if (it->second == true) {
      // This is a noise frame
      num_noise++;
      noise_sum.AddVec(1.0, cur_frame);
      noise_var.AddVecVec(1.0, cur_frame, cur_frame);  
    } else {
      // This is a speech frame
      num_speech++;
      speech_sum.AddVec(1.0, cur_frame);
      speech_var.AddVecVec(1.0, cur_frame, cur_frame);
    }
  }

  // See paper for the math for this estimation method
  Matrix<BaseFloat> K(2*dim, 2*dim);
  Vector<BaseFloat> Q(2*dim);
  
  // Computing the matrix K
  SubMatrix<BaseFloat> K_11(K, 0, dim, 0, dim), K_12(K, 0, dim, dim, dim),
    K_21(K, dim, dim, 0, dim), K_22(K, dim, dim, dim, dim);
  K_11 = nvector_stats_.Lambda_s_;
  K_11.Scale(1 + num_speech);
  K_12.AddMatMat(-1, nvector_stats_.Lambda_s_, kNoTrans, nvector_stats_.B, kNoTrans, 0);
  K_21.AddMatMat(-1, nvector_stats_.B_, kTrans, nvector_stats_.Lambda_s_, kNoTrans, 0);
  K_22 = nvector_stats_.Lambda_n_;
  K_22.Scale(1 + num_noise);
  Matrix<BaseFloat> temp(dim, dim);
  temp.AddMatMat(1, nvector_stats_.B, kTrans, nvector_stats_.Lambda_s_, kNoTrans, 0);
  K_22.AddMatMat(1, temp, kNoTrans, nvector_stats_.B_, kNoTrans, 1);
  
  // Computing the vector Q
  SubVector<BaseFloat> Q_1(Q, 0, dim), Q_2(Q, dim, dim);
  Vector<BaseFloat> temp = nvector_stats_.a_;
  temp.AddVec(1, speech_sum);
  Q_1.AddMatVec(0, nvector_stats_.Lambda_s_, kNoTrans, temp);
  temp = nvector_stats_.mu_n_;
  temp.AddVec(1, noise_sum);
  Q_2.AddMatVec(0, nvector_stats_.Lambda_n_, kNoTrans, temp);
  temp.AddMatVec(0, nvector_stats_.Lambda_s_, kNoTrans, nvector_stats_.a_);
  Q_2.AddMatVec(1, nvector_stats_.B_, kTrans, temp);
  
  // Compute the nvector from K and Q
  K.Invert();
  current_nvector_.AddMatVec(0, K, kNoTrans, Q);
  nvectors_history_.push_back(current_nvector_);
}

OnlineNvectorFeature::~OnlineNvectorFeature() {
  PrintDiagnostics();
  // Delete objects owned here.
  for (size_t i = 0; i < to_delete_.size(); i++)
    delete to_delete_[i];
  for (size_t i = 0; i < nvectors_history_.size(); i++)
    delete nvectors_history_[i];
}


OnlineSilenceDetection::OnlineSilenceDetection(
    const TransitionModel &trans_model,
    const OnlineSilenceDetectionConfig &config,
    int32 frame_subsampling_factor):
    trans_model_(trans_model), config_(config),
    frame_subsampling_factor_(frame_subsampling_factor),
    num_frames_output_and_correct_(0) {
  KALDI_ASSERT(frame_subsampling_factor_ >= 1);
  std::vector<int32> silence_phones;
  SplitStringToIntegers(config.silence_phones_str, ":,", false,
                        &silence_phones);
  for (size_t i = 0; i < silence_phones.size(); i++)
    silence_phones_.insert(silence_phones[i]);
}


template <typename FST>
void OnlineSilenceDetection::DecodeNextChunk(
    const LatticeFasterOnlineDecoderTpl<FST> &decoder) {
  int32 num_frames_decoded = decoder.NumFramesDecoded(),
      num_frames_prev = frame_info_.size();
  // Since we do not recompute silence decisions for previously
  // decoded frames, the num_frames_decoded must be equal to
  // num_frames_prev
  KALDI_ASSERT(num_frames_decoded == num_frames_prev);

  if (num_frames_decoded == 0)
    return;
  int32 frame = num_frames_decoded - 1;
  bool use_final_probs = false;
  typename LatticeFasterOnlineDecoderTpl<FST>::BestPathIterator iter =
      decoder.BestPathEnd(use_final_probs, NULL);
  while (frame >= 0) {
    LatticeArc arc;
    arc.ilabel = 0;
    while (arc.ilabel == 0)  // the while loop skips over input-epsilons
      iter = decoder.TraceBackBestPath(iter, &arc);
    // note, the iter.frame values are slightly unintuitively defined,
    // they are one less than you might expect.
    KALDI_ASSERT(iter.frame == frame - 1);

    if (frame_info_[frame].token == iter.tok) {
      // we know that the traceback from this point back will be identical, so
      // no point tracing back further.  Note: we are comparing memory addresses
      // of tokens of the decoder; this guarantees it's the same exact token
      // because tokens, once allocated on a frame, are only deleted, never
      // reallocated for that frame.
      break;
    }

    frame_info_[frame].token = iter.tok;
    frame_info_[frame].transition_id = arc.ilabel;
    frame--;
    // leave frame_info_.current_weight at zero for now (as set in the
    // constructor), reflecting that we haven't already output a weight for that
    // frame.
  }
}

// Instantiate the template OnlineSilenceDetection::DecodeNextChunk().
template
void OnlineSilenceDetection::DecodeNextChunk<fst::Fst<fst::StdArc> >(
    const LatticeFasterOnlineDecoderTpl<fst::Fst<fst::StdArc> > &decoder);
template
void OnlineSilenceDetection::DecodeNextChunk<fst::GrammarFst>(
    const LatticeFasterOnlineDecoderTpl<fst::GrammarFst> &decoder);

void OnlineSilenceDetection::GetSilenceDecisions(
    int32 num_frames_ready, int32 first_decoder_frame,
    std::vector<std::pair<int32, bool> > *silence_frames) {
  // num_frames_ready is at the feature frame-rate, most of the code
  // in this function is at the decoder frame-rate.
  // round up, so we are sure to get weights for at least the frame
  // 'num_frames_ready - 1', and maybe one or two frames afterward.
  KALDI_ASSERT(num_frames_ready > first_decoder_frame || num_frames_ready == 0);
  int32 fs = frame_subsampling_factor_,
  num_decoder_frames_ready = (num_frames_ready - first_decoder_frame + fs - 1) / fs;

  silence_frames->clear();

  int32 prev_num_frames_processed = frame_info_.size();
  if (frame_info_.size() < static_cast<size_t>(num_decoder_frames_ready))
    frame_info_.resize(num_decoder_frames_ready);

  // We start from the last computed frame. This is different from the online
  // silence weighting done for ivectors, where we go 100 frames in history
  // to recompute silence decisions. This is because we are using a Bayesian
  // model for n-vector estimation.
  int32 begin_frame = prev_num_frames_processed,
      frames_out = static_cast<int32>(frame_info_.size()) - begin_frame;
  // frames_out is the number of frames we will output.
  KALDI_ASSERT(frames_out >= 0);
  std::vector<bool> frame_decisions(frames_out, false);
  // we will set frame_decisions to the value true for silence frames and
  // for transition-ids that repeat with duration > max_state_duration.

  // First treat some special cases.
  if (frames_out == 0)  // Nothing to output.
    return;
  if (frame_info_[begin_frame].transition_id == -1) {
    // We do not have any traceback at all within the frames we are to output...
    bool decision = (begin_frame == 0 ? true :
                        frame_info_[begin_frame - 1].silence_decision);
    for (int32 offset = 0; offset < frames_out; offset++)
      frame_decisions[offset] = decision;
  } else {
    int32 current_run_start_offset = 0;
    for (int32 offset = 0; offset < frames_out; offset++) {
      int32 frame = begin_frame + offset;
      int32 transition_id = frame_info_[frame].transition_id;
      if (transition_id == -1) {
        // this frame does not yet have a decoder traceback, so just
        // duplicate the silence/non-silence status of the most recent
        // frame we have a traceback for (probably a reasonable guess).
        frame_decisions[offset] = frame_decisions[offset - 1];
      } else {
        int32 phone = trans_model_.TransitionIdToPhone(transition_id);
        bool is_silence = (silence_phones_.count(phone) != 0);
        if (is_silence)
          frame_decisions[offset] = true;
        // now deal with max-duration issues.
        if (max_state_duration > 0 &&
            (offset + 1 == frames_out ||
             transition_id != frame_info_[frame + 1].transition_id)) {
          // If this is the last frame of a run...
          int32 run_length = offset - current_run_start_offset + 1;
          if (run_length >= max_state_duration) {
            // treat runs of the same transition-id longer than the max, as
            // silence, even if they were not silence.
            for (int32 offset2 = current_run_start_offset;
                 offset2 <= offset; offset2++)
              frame_decisions[offset2] = true;
          }
          if (offset + 1 < frames_out)
            current_run_start_offset = offset + 1;
        }
      }
    }
  }
  // Now commit the stats...
  for (int32 offset = 0; offset < frames_out; offset++) {
    int32 frame = begin_frame + offset;
    frame_info_[frame].silence_decision = frame_decisions[offset];
    // Even if the delta-weight is zero for the last frame, we provide it,
    // because the identity of the most recent frame with a weight is used in
    // some debugging/checking code.
    for(int32 i = 0; i < frame_subsampling_factor_; i++) {
      int32 input_frame = first_decoder_frame + (frame * frame_subsampling_factor_) + i;
      silence_frames->push_back(std::make_pair(input_frame, frame_decisions[i]));
    }
  }
}

}  // namespace kaldi
