// online2/online-nvector-feature.h

// Copyright 2020   Johns Hopkins University (author: Desh Raj)

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


#ifndef KALDI_ONLINE2_ONLINE_NVECTOR_FEATURE_H_
#define KALDI_ONLINE2_ONLINE_NVECTOR_FEATURE_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "itf/online-feature-itf.h"
#include "gmm/diag-gmm.h"
#include "feat/online-feature.h"
#include "decoder/lattice-faster-online-decoder.h"

namespace kaldi {
/// @addtogroup  onlinefeat OnlineFeatureExtraction
/// @{

/// @file
/// This file contains code for online iVector extraction in a form compatible
/// with OnlineFeatureInterface.  It's used in online-nnet2-feature-pipeline.h.

/// This class includes configuration variables relating to the online iVector
/// extraction, but not including configuration for the "base feature",
/// i.e. MFCC/PLP/filterbank, which is an input to this feature.  This
/// configuration class can be used from the command line, but before giving it
/// to the code we create a config class called
/// OnlineIvectorExtractionInfo which contains the actual configuration
/// classes as well as various objects that are needed.  The principle is that
/// any code should be callable from other code, so we didn't want to force
/// configuration classes to be read from disk.
struct OnlineNvectorExtractionConfig {
  
  std::string noise_prior_rxfilename;  // reads type OnlineNvectorEstimationParams

  int32 nvector_period;  // How frequently we re-estimate n-vectors.

  // max_remembered_frames is the largest number of frames it will remember
  // between utterances of the same speaker; this affects the output of
  // GetAdaptationState(), and has the effect of limiting the number of frames
  // of both the CMVN stats and the noise vector stats.  Setting this to a smaller
  // value means the adaptation is less constrained by previous utterances
  // (assuming you provided info from a previous utterance of the same speaker
  // by calling SetAdaptationState()).
  BaseFloat max_remembered_frames;

  OnlineNvectorExtractionConfig(): nvector_period(10),
                                   max_remembered_frames(1000) { }

  void Register(OptionsItf *opts) {
    opts->Register("noise-prior", &noise_prior_rxfilename,
                   "Filename of prior parameters estimated from training data.");
    opts->Register("nvector-period", &nvector_period, "Frequency with which "
                   "we extract noise vectors for neural network adaptation");
    opts->Register("max-remembered-frames", &max_remembered_frames, "The maximum "
                   "number of frames of adaptation history that we carry through "
                   "to later utterances of the same speaker (having a finite "
                   "number allows the speaker adaptation state to change over "
                   "time).");  
  }
};

// forward declaration
class OnlineNvectorFeature;

/// OnlineNvectorEstimationParams is a class that stores the parameters
/// required to estimate the online noise vectors. It also stores the 
/// prior parameters required for initial estimattion of noise vectors.
/// Further, it contains a method to compute the prior parameters
/// given the statistics of the training data.
///
/// At test time for online decoding, the OnlineNvectorFeature class
/// initializes its own copy of OnlineNvectorEstimationParams when
/// SetAdaptationState() function is called, and then updates the
/// parameters of the copy using an E-M like procedure.

class OnlineNvectorEstimationParams {
 friend class OnlineNvectorFeature;

 public:
  OnlineNvectorEstimationParams():
    r_s_(1), r_n_(1) { }

  explicit OnlineNvectorEstimationParams(const OnlineNvectorEstimationParams &other):
    mu_n_(other.mu_n_),
    a_(other.a_),
    B_(other.B_),
    Lambda_n_(other.Lambda_n_),
    Lambda_s_(other.Lambda_s_),
    r_s_(other.r_s_),
    r_n_(other.r_n_) {
  };

  // Assignment operator used for GetAdaptationState() and SetAdaptationState() methods
  OnlineNvectorEstimationParams &operator = (const OnlineNvectorEstimationParams &other) {
    return *this;
  }

  /// Takes the mean and covariance matrix computed from the
  /// training data and estimates the prior parameters.
  void EstimatePriorParameters(const VectorBase<BaseFloat> &mean,
                               const SpMatrix<BaseFloat> &covariance,
                               int32 dim);

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

 protected:
  Vector<BaseFloat> mu_n_;  // mean of noise vectors.
  Vector<BaseFloat> a_;  // shift factor for mean of speech vectors.
  Matrix<BaseFloat> B_;  // scale factor for mean of speech vectors.
  Matrix<BaseFloat> Lambda_n_; // precision matrix for noise.
  Matrix<BaseFloat> Lambda_s_; // precision matrix for speech.
  double r_s_; // scaling factor for speech.
  double r_n_; // scaling factor for noise.

};

// This struct contains various things that are needed by the 
// OnlineNvectorEstimationParams class for extracting noise
// vectors.
struct OnlineNvectorExtractionInfo {

  OnlineNvectorEstimationParams noise_prior;

  int32 nvector_period;
  BaseFloat max_remembered_frames;

  OnlineNvectorExtractionInfo(const OnlineNvectorExtractionConfig &config);

  void Init(const OnlineNvectorExtractionConfig &config);

  // This constructor creates a version of this object where everything
  // is empty or zero.
  OnlineNvectorExtractionInfo();

  void Check() const;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineNvectorExtractionInfo);
};


/// OnlineNvectorFeature is an online feature-extraction class that's responsible
/// for extracting noise vectors from raw features such as MFCC, PLP or filterbank.
/// Internally it processes the raw features using two different pipelines, one
/// online-CMVN+splice+LDA, and one just splice+LDA. It gets GMM posteriors from
/// the CMVN-normalized features, and with those and the unnormalized features
/// it obtains noise vectors.

class OnlineNvectorFeature: public OnlineFeatureInterface {
 public:
  /// Constructor.  base_feature is for example raw MFCC or PLP or filterbanks.
  /// "info" contains all the configuration information as well as
  /// things like the noise prior that we won't be modifying.
  /// Caution: the class keeps a const reference to "info", so don't
  /// delete it while this class or others copied from it still exist.
  explicit OnlineNvectorFeature(const OnlineNvectorExtractionInfo &info,
                                OnlineFeatureInterface *base_feature);


  // Member functions from OnlineFeatureInterface:

  /// Dim() will return the noise vector dimension.
  virtual int32 Dim() const;
  virtual bool IsLastFrame(int32 frame) const;
  virtual int32 NumFramesReady() const;
  virtual BaseFloat FrameShiftInSeconds() const;
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  /// Set the adaptation state to a particular value, e.g. reflecting previous
  /// utterances of the same speaker; this will generally be called after
  /// constructing a new instance of this class.
  void SetAdaptationState(
      const OnlineNvectorEstimationParams &adaptation_state);


  /// Get the adaptation state; you may want to call this before destroying this
  /// object, to get adaptation state that can be used to improve decoding of
  /// later utterances of this speaker.
  void GetAdaptationState(
      OnlineNvectorEstimationParams *adaptation_state) const;

  virtual ~OnlineNvectorFeature();

  // This function updates current_nvector_  (which is our present estimate)
  // of the  current value for the n-vector, after a new chunk of 
  // data is seen. It takes as argument the silence decisions made by the 
  // GmmDecoder.
  void UpdateNvector(
      const std::vector<std::pair<int32, bool> > &silence_frames);

  // This function updates the scaling parameters r_s and r_n of the 
  // noise estimation model. This is done by maximizing the EM
  // objective. The derivation is not shown here.
  void UpdateScalingParams(
      const std::vector<std::pair<int32, bool> > &silence_frames);

 private:

  void PrintDiagnostics() const;

  const OnlineNvectorExtractionInfo &info_;

  OnlineFeatureInterface *base_;  // The feature this is built on top of
                                  // (e.g. MFCC); not owned here

  // the following is the pointers to OnlineFeatureInterface objects that are
  // owned here and which we need to delete.
  std::vector<OnlineFeatureInterface*> to_delete_;

  /// the noise vector estimation stats
  OnlineNvectorEstimationParams nvector_stats_;

  /// num_frames_stats_ is the number of frames of data we have already
  /// accumulated from this utterance and put in nvector_stats_.  Each frame t <
  /// num_frames_stats_ is in the stats.
  int32 num_frames_stats_;

  /// Most recently estimated noise vector, will have been
  /// estimated at the greatest time t where t <= num_frames_stats_ and
  /// t % info_.nvector_period == 0.
  Vector<BaseFloat> current_nvector_;

  /// if info_.use_most_recent_nvector == false, we need to store
  /// the n-vector we estimated each info_.nvector_period frames so that
  /// GetFrame() can return the noise vector that was active on that frame.
  /// nvectors_history_[i] contains the noise vector we estimated on
  /// frame t = i * info_.nvector_period.
  std::vector<Vector<BaseFloat>* > nvectors_history_;

};


struct OnlineSilenceDetectionConfig {
  std::string silence_phones_str;
  int32 max_state_duration;

  bool Active() const {
    return !silence_phones_str.empty();
  }

  OnlineSilenceDetectionConfig(): max_state_duration(-1) { }

  void Register(OptionsItf *opts) {
    opts->Register("silence-phones", &silence_phones_str, "(RE weighting in "
                   "noise vector estimation for online decoding) List of integer ids of "
                   "silence phones, separated by colons (or commas).");
  }
  // e.g. prefix = "ivector-silence-weighting"
  void RegisterWithPrefix(std::string prefix, OptionsItf *opts) {
    ParseOptions po_prefix(prefix, opts);
    this->Register(&po_prefix);
  }
};


// This class is responsible for performing speech/silence frame classification
// which is used for computing means for the online estimation of noise
// vectors. 
class OnlineSilenceDetection {
 public:
  // Note: you would initialize a new copy of this object for each new
  // utterance.
  // The frame-subsampling-factor is used for newer nnet3 models, especially
  // chain models, when the frame-rate of the decoder is different from the
  // frame-rate of the input features.  E.g. you might set it to 3 for such
  // models.

  OnlineSilenceDetection(const TransitionModel &trans_model,
                         const OnlineSilenceDetectionConfig &config,
                         int32 frame_subsampling_factor = 1);

  bool Active() const { return config_.Active(); }

  // This should be called before GetSilenceDecisions, so this class knows about the
  // traceback info from the decoder.  It records the traceback information from
  // the decoder using its BestPathEnd() and related functions.
  // It will be instantiated for FST == fst::Fst<fst::StdArc> and fst::GrammarFst.
  template <typename FST>
  void DecodeNextChunk(const LatticeFasterOnlineDecoderTpl<FST> &decoder);

  // This function outputs speech/silence decision for every frame in the utterance
  // and the output format is (frame-index, true/false), where true/false refers to
  // silence/speech frames respectively.
  //
  // The num_frames_ready argument is the number of frames available at
  // the input (or equivalently, output) of the online n-vector feature in the
  // feature pipeline from the stream start. It may be more than the currently
  // available decoder traceback.
  //
  // The first_decoder_frame is the offset from the start of the stream in
  // pipeline frames when decoder was restarted last time. We do not change
  // weight for the frames earlier than first_decoder_frame. Set it to 0 in
  // case of compilation error to reproduce the previous behavior or for a
  // single utterance decoding.
  //
  // How many frames of decisions it outputs depends on how much "num_frames_ready"
  // increased since last time we called this function, and whether the decoder
  // traceback changed.  You must call this function with "num_frames_ready"
  // arguments that only increase, not decrease, with time.  You would provide
  // this output to class OnlineNvectorFeature by calling its function
  // UpdateNvectors with the output.
  //
  // Returned frame-index is in pipeline frames from the pipeline start.
  void GetSilenceDecisions(
      int32 num_frames_ready, int32 first_decoder_frame,
      std::vector<std::pair<int32, bool> > *silence_frames);

  // A method for backward compatibility, same as above, but for a single
  // utterance.
  void GetSilenceDecisions(
      int32 num_frames_ready,
      std::vector<std::pair<int32, bool> > *silence_frames) {
    GetSilenceDecisions(num_frames_ready, 0, silence_frames);
  }

 private:
  const TransitionModel &trans_model_;
  const OnlineSilenceDetectionConfig &config_;

  int32 frame_subsampling_factor_;

  unordered_set<int32> silence_phones_;
  int32 max_state_duration_;

  struct FrameInfo {
    // The only reason we need the token pointer is to know far back we have to
    // trace before the traceback is the same as what we previously traced back.
    void *token;
    int32 transition_id;
    bool silence_decision;
    FrameInfo(): token(NULL), transition_id(-1), silence_decision(true) {}
  };

  // This contains information about any previously computed traceback;
  // when the traceback changes we use this variable to compare it with the
  // previous traceback.
  // It's indexed at the frame-rate of the decoder (may be different
  // by 'frame_subsampling_factor_' from the frame-rate of the features.
  std::vector<FrameInfo> frame_info_;

};


/// @} End of "addtogroup onlinefeat"
}  // namespace kaldi

#endif  // KALDI_ONLINE2_ONLINE_NVECTOR_FEATURE_H_
