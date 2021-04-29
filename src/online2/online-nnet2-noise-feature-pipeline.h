// online2/online-nnet2-noise-feature-pipeline.h

// Copyright 2013-2014   Johns Hopkins University (author: Daniel Povey)
//           2020        Johns Hopkins University (author: Desh Raj)

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


#ifndef KALDI_ONLINE2_ONLINE_NNET2_NOISE_FEATURE_PIPELINE_H_
#define KALDI_ONLINE2_ONLINE_NNET2_NOISE_FEATURE_PIPELINE_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "feat/online-feature.h"
#include "feat/pitch-functions.h"
#include "online2/online-nvector-feature.h"

namespace kaldi {
/// @addtogroup  onlinefeat OnlineFeatureExtraction
/// @{

/// @file
/// This file contains a version of the feature-extraction pipeline specialized 
/// for use in neural network decoding with noise vectors. We use a Bayesian
/// model of the means of speech and silence frames in the utterance, and
/// these are used as additional input to the neural network, in addition to
/// a window of several frames of spliced raw features (MFCC, PLP, or 
/// filterbanks). Noise vector computation requires a speech activity detection
/// system, for which we use GMM posteriors with online cepstral mean (and 
/// optionally variance) normalization, but the means are computed with a 
/// non-mean-normalized version of the features. The idea behind this is the
/// same as that for online ivector extraction, i.e., we want the noise vectors 
/// to learn the mean offset, but we want the posteriors to be somewhat invariant
/// to mean offsets.
///
/// Most of the logic for the actual noise vector estimation is in \ref
/// online-noise-vector-feature.h, this header contains mostly glue.
///
/// Although the name of this header mentions nnet2, actually the code is
/// used in the online decoding with nnet3 also.
///
/// The class OnlineNnet2NoiseFeaturePipeline also has a support to optionally
/// append pitch features and to apply OnlineCmvn on nnet3 input.
/// We pass the unnormalized base_features for noise vector computation,
/// the OnlineCmvn for noise vector computation is handled elsewhere.


/// This configuration class is to set up OnlineNnet2NoiseFeaturePipelineInfo, which
/// in turn is the configuration class for OnlineNnet2NoiseFeaturePipeline.
/// Instead of taking the options for the parts of the feature pipeline
/// directly, it reads in the names of configuration classes.
struct OnlineNnet2NoiseFeaturePipelineConfig {
  std::string feature_type;  // "plp" or "mfcc" or "fbank"
  std::string mfcc_config;

  // Note: if we do add pitch, it will not be added to the features we give to
  // the noise vector computation but only to the features we give to the neural
  // network, after the base features but before the noise vector.
  bool add_pitch;

  // the following contains the type of options that you could give to
  // compute-and-process-kaldi-pitch-feats.
  std::string online_pitch_config;

  // The configuration variables in nvector_extraction_config relate to the
  // noise vector extractor and options related to it, see type
  // OnlineNvectorExtractionConfig.
  std::string nvector_extraction_config;
  std::string silence_phones_str;

  OnlineNnet2NoiseFeaturePipelineConfig():
      feature_type("mfcc"), add_pitch(false) { }


  void Register(OptionsItf *opts) {
    opts->Register("feature-type", &feature_type,
                   "Base feature type [mfcc, plp, fbank]");
    opts->Register("mfcc-config", &mfcc_config, "Configuration file for "
                   "MFCC features (e.g. conf/mfcc.conf)");
    opts->Register("nvector-extraction-config", &nvector_extraction_config,
                   "Configuration file for online noise vector extraction, "
                   "see class OnlineNvectorExtractionConfig in the code");
    opts->Register("add-pitch", &add_pitch, "Append pitch features to raw "
                   "MFCC/PLP/filterbank features [but not for noise vector computation]");
    opts->Register("online-pitch-config", &online_pitch_config, "Configuration "
                   "file for online pitch features, if --add-pitch=true (e.g. "
                   "conf/online_pitch.conf)");
    opts->Register("silence-phones", &silence_phones_str, "(RE weighting in "
                   "noise vector estimation for online decoding) List of integer ids of "
                   "silence phones, separated by colons (or commas).");
  }
};


/// This class is responsible for storing configuration variables, objects and
/// options for OnlineNnet2NoiseFeaturePipeline. This class does not register
/// options on the command line; instead, it is initialized from class
/// OnlineNnet2NoiseFeaturePipelineConfig which reads the options from the command
/// line.  The reason for structuring it this way is to make it easier to
/// configure from code as well as from the command line, as well as for easier
/// multithreaded operation.
struct OnlineNnet2NoiseFeaturePipelineInfo {
  OnlineNnet2NoiseFeaturePipelineInfo():
      feature_type("mfcc"), add_pitch(false) { }

  OnlineNnet2NoiseFeaturePipelineInfo(
      const OnlineNnet2NoiseFeaturePipelineConfig &config);

  BaseFloat FrameShiftInSeconds() const;

  std::string feature_type; /// "mfcc" or "plp" or "fbank"

  MfccOptions mfcc_opts;  /// options for MFCC computation,
                          /// if feature_type == "mfcc"
  PlpOptions plp_opts;    /// Options for PLP computation, if feature_type == "plp"
  FbankOptions fbank_opts;  /// Options for filterbank computation, if
                            /// feature_type == "fbank"

  bool add_pitch;
  PitchExtractionOptions pitch_opts;  /// Options for pitch extraction, if done.
  ProcessPitchOptions pitch_process_opts;  /// Options for pitch post-processing

  /// If the user specified --nvector-extraction-config, we assume we're using
  /// noise vectors as an extra input to the neural net.  Actually, we don't
  /// anticipate running this setup without noise vectors.
  bool use_nvectors;
  OnlineNvectorExtractionInfo nvector_extraction_info;

  std::string silence_phones_str;
  int32 max_state_duration;

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineNnet2NoiseFeaturePipelineInfo);
};



/// OnlineNnet2NoiseFeaturePipeline is a class that's responsible for putting
/// together the various parts of the feature-processing pipeline for neural
/// networks, in an online setting.  The recipe here does not include fMLLR;
/// instead, it assumes we're giving raw features such as MFCC or PLP or
/// filterbank (with no CMVN) to the neural network, and augmenting these with
/// a noise vector that describes the speaker/channel characteristics.  The noise
/// vector is extracted using class OnlineNoiseVectorFeature (see that class for
/// more info on how it's done).
/// No splicing is currently done in this code, as we're currently only supporting
/// the nnet2 neural network in which the splicing is done inside the network.
/// Probably our strategy for nnet1 network conversion would be to convert to nnet2
/// and just add layers to do the splicing.
class OnlineNnet2NoiseFeaturePipeline: public OnlineFeatureInterface {
 public:
  /// Constructor from the "info" object.  After calling this for a
  /// non-initial utterance of a speaker, you may want to call
  /// SetAdaptationState().
  explicit OnlineNnet2NoiseFeaturePipeline(
      const OnlineNnet2NoiseFeaturePipelineInfo &info);

  /// Member functions from OnlineFeatureInterface:

  /// Dim() will return the base-feature dimension (e.g. 13 for normal MFCC);
  /// plus the pitch-feature dimension (e.g. 3), if used; plus the noise vector
  /// dimension.  Any frame-splicing happens inside the neural-network code.
  virtual int32 Dim() const;

  virtual bool IsLastFrame(int32 frame) const;
  virtual int32 NumFramesReady() const;
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  /// Set the adaptation state to a particular value, e.g. reflecting previous
  /// utterances of the same speaker; this will generally be called after
  /// Copy().
  void SetAdaptationState(
      const OnlineNvectorEstimationParams &adaptation_state);

  /// Get the adaptation state; you may want to call this before destroying this
  /// object, to get adaptation state that can be used to improve decoding of
  /// later utterances of this speaker.  You might not want to do this, though,
  /// if you have reason to believe that something went wrong in the recognition
  /// (e.g., low confidence).
  void GetAdaptationState(
      OnlineNvectorEstimationParams *adaptation_state) const;

  /// Accept more data to process.  It won't actually process it until you call
  /// GetFrame() [probably indirectly via (decoder).AdvanceDecoding()], when you
  /// call this function it will just copy it).  sampling_rate is necessary just
  /// to assert it equals what's in the config.
  void AcceptWaveform(BaseFloat sampling_rate,
                      const VectorBase<BaseFloat> &waveform);

  BaseFloat FrameShiftInSeconds() const { return info_.FrameShiftInSeconds(); }

  /// If you call InputFinished(), it tells the class you won't be providing any
  /// more waveform.  This will help flush out the last few frames of delta or
  /// LDA features, and finalize the pitch features (making them more
  /// accurate)... although since in neural-net decoding we don't anticipate
  /// rescoring the lattices, this may not be much of an issue.
  void InputFinished();

  /// This function returns the nvector-extracting part of the feature pipeline.
  /// The pointer ownership is retained by this object and not transferred to the 
  /// caller. This function is used in nnet3
  OnlineNvectorFeature *NvectorFeature() {
    return nvector_feature_;
  }

  /// A const accessor for the noise vector extractor. Returns NULL if noise
  /// vectors are not being used.
  const OnlineNvectorFeature *NvectorFeature() const {
    return nvector_feature_;
  }

  /// This function returns the part of the feature pipeline that would be given
  /// as the primary (non-nvector) input to the neural network in nnet3
  /// applications.
  OnlineFeatureInterface *InputFeature() {
    return nnet3_feature_;
  }

  virtual ~OnlineNnet2NoiseFeaturePipeline();

 private:
  const OnlineNnet2NoiseFeaturePipelineInfo &info_;

  OnlineBaseFeature *base_feature_;    /// MFCC/PLP/filterbank

  OnlinePitchFeature *pitch_;          /// Raw pitch, if used
  OnlineProcessPitch *pitch_feature_;  /// Processed pitch, if pitch used.

  /// feature_plus_optional_pitch_ is the base_feature_ appended (OnlineAppendFeature)
  /// with pitch_feature_, if used; otherwise, points to the same address as
  /// base_feature_.
  OnlineFeatureInterface *feature_plus_optional_pitch_;

  OnlineNvectorFeature *nvector_feature_;  /// noise vector feature, if used.

  /// Part of the feature pipeline that would be given as the primary
  /// (non-noise-vector) input to the neural network in nnet3 applications.
  /// This pointer is returned by InputFeature().
  OnlineFeatureInterface *nnet3_feature_;

  /// final_feature_ is feature_plus_optional_pitch_ appended
  /// (OnlineAppendFeature) with noise_vector_feature_
  OnlineFeatureInterface *final_feature_;

  /// we cache the feature dimension, to save time when calling Dim().
  int32 dim_;
};


/// @} End of "addtogroup onlinefeat"
}  // namespace kaldi



#endif  // KALDI_ONLINE2_ONLINE_NNET2_NOISE_FEATURE_PIPELINE_H_
