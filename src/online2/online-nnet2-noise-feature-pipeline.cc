// online2/online-nnet2-noise-feature-pipeline.cc

// Copyright    2013  Johns Hopkins University (author: Daniel Povey)
//              2020  Johns Hopkins University (author: Desh Raj)

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

#include "online2/online-nnet2-noise-feature-pipeline.h"
#include "transform/cmvn.h"

namespace kaldi {

OnlineNnet2NoiseFeaturePipelineInfo::OnlineNnet2NoiseFeaturePipelineInfo(
    const OnlineNnet2NoiseFeaturePipelineConfig &config) {
  if (config.feature_type == "mfcc" || config.feature_type == "plp" ||
      config.feature_type == "fbank") {
    feature_type = config.feature_type;
  } else {
    KALDI_ERR << "Invalid feature type: " << config.feature_type << ". "
              << "Supported feature types: mfcc, plp, fbank.";
  }

  if (config.mfcc_config != "") {
    ReadConfigFromFile(config.mfcc_config, &mfcc_opts);
    if (feature_type != "mfcc")
      KALDI_WARN << "--mfcc-hires-config option has no effect "
                 << "since feature type is set to " << feature_type << ".";
  }  // else use the defaults.

  add_pitch = config.add_pitch;

  if (config.online_pitch_config != "") {
    ReadConfigsFromFile(config.online_pitch_config,
                        &pitch_opts,
                        &pitch_process_opts);
    if (!add_pitch)
      KALDI_WARN << "--online-pitch-config option has no effect "
                 << "since you did not supply --add-pitch option.";
  }  // else use the defaults.

  use_cmvn = (config.cmvn_config != "");
  if (use_cmvn) {
    ReadConfigFromFile(config.cmvn_config, &cmvn_opts);
    global_cmvn_stats_rxfilename = config.global_cmvn_stats_rxfilename;
    if (global_cmvn_stats_rxfilename == "")
      KALDI_ERR << "--global-cmvn-stats option is required "
                << " when --cmvn-config is specified.";
  }
  
  if (config.nvector_extraction_config != "") {
    use_nvectors = true;
    OnlineNvectorExtractionConfig nvector_extraction_opts;
    ReadConfigFromFile(config.nvector_extraction_config,
                       &nvector_extraction_opts);
    nvector_extraction_info.Init(nvector_extraction_opts);


  } else {
    use_nvectors = false;
  }
}


/// The main feature extraction pipeline is constructed in this constructor.
OnlineNnet2NoiseFeaturePipeline::OnlineNnet2NoiseFeaturePipeline(
    const OnlineNnet2NoiseFeaturePipelineInfo &info):
    info_(info), base_feature_(NULL),
    pitch_(NULL), pitch_feature_(NULL),
    cmvn_feature_(NULL),
    feature_plus_optional_pitch_(NULL),
    feature_plus_optional_cmvn_(NULL),
    nvector_feature_(NULL),
    nnet3_feature_(NULL),
    final_feature_(NULL) {

  if (info_.feature_type == "mfcc") {
    base_feature_ = new OnlineMfcc(info_.mfcc_opts);
  } else if (info_.feature_type == "plp") {
    base_feature_ = new OnlinePlp(info_.plp_opts);
  } else if (info_.feature_type == "fbank") {
    base_feature_ = new OnlineFbank(info_.fbank_opts);
  } else {
    KALDI_ERR << "Code error: invalid feature type " << info_.feature_type;
  }

  if (info_.add_pitch) {
    pitch_ = new OnlinePitchFeature(info_.pitch_opts);
    pitch_feature_ = new OnlineProcessPitch(info_.pitch_process_opts,
                                            pitch_);
    feature_plus_optional_pitch_ = new OnlineAppendFeature(base_feature_,
                                                           pitch_feature_);
  } else {
    feature_plus_optional_pitch_ = base_feature_;
  }

  if (info_.use_cmvn) {
    KALDI_ASSERT(info.global_cmvn_stats_rxfilename != "");
    ReadKaldiObject(info.global_cmvn_stats_rxfilename, &global_cmvn_stats_);
    OnlineCmvnState initial_state(global_cmvn_stats_);
    cmvn_feature_ = new OnlineCmvn(info_.cmvn_opts, initial_state,
        feature_plus_optional_pitch_);
    feature_plus_optional_cmvn_ = cmvn_feature_;
  } else {
    feature_plus_optional_cmvn_ = feature_plus_optional_pitch_;
  }

  if (info.use_nvectors) {
    nnet3_feature_ = feature_plus_optional_cmvn_;
    // Note: the noise vector extractor OnlineNvectorFeature gets 'base_feautre_'
    // without cmvn (the online cmvn is applied inside the class)
    nvector_feature_ = new OnlineNvectorFeature(info_.nvector_extraction_info,
                                                base_feature_);
    final_feature_ = new OnlineAppendFeature(feature_plus_optional_cmvn_,
                                             nvector_feature_);
  } else {
    nnet3_feature_ = feature_plus_optional_cmvn_;
    final_feature_ = feature_plus_optional_cmvn_;
  }
  dim_ = final_feature_->Dim();
}
/// ^-^


int32 OnlineNnet2NoiseFeaturePipeline::Dim() const { return dim_; }

bool OnlineNnet2NoiseFeaturePipeline::IsLastFrame(int32 frame) const {
  return final_feature_->IsLastFrame(frame);
}

int32 OnlineNnet2NoiseFeaturePipeline::NumFramesReady() const {
  return final_feature_->NumFramesReady();
}

void OnlineNnet2NoiseFeaturePipeline::GetFrame(int32 frame,
                                          VectorBase<BaseFloat> *feat) {
  return final_feature_->GetFrame(frame, feat);
}

void OnlineNnet2NoiseFeaturePipeline::SetAdaptationState(
    const OnlineNvectorEstimationParams &adaptation_state) {
  if (info_.use_nvectors) {
    nvector_feature_->SetAdaptationState(adaptation_state);
  }
  // else silently do nothing, as there is nothing to do.
}

void OnlineNnet2NoiseFeaturePipeline::GetAdaptationState(
    OnlineNvectorEstimationParams *adaptation_state) const {
  if (info_.use_nvectors) {
    nvector_feature_->GetAdaptationState(adaptation_state);
  }
  // else silently do nothing, as there is nothing to do.
}

void OnlineNnet2NoiseFeaturePipeline::SetCmvnState(
    const OnlineCmvnState &cmvn_state) {
  if (NULL != cmvn_feature_)
    cmvn_feature_->SetState(cmvn_state);
}

void OnlineNnet2NoiseFeaturePipeline::GetCmvnState(
    OnlineCmvnState *cmvn_state) {
  if (NULL != cmvn_feature_) {
    int32 frame = cmvn_feature_->NumFramesReady() - 1;
    // the following call will crash if no frames are ready.
    cmvn_feature_->GetState(frame, cmvn_state);
  }
}


OnlineNnet2NoiseFeaturePipeline::~OnlineNnet2NoiseFeaturePipeline() {
  // Note: the delete command only deletes pointers that are non-NULL.  Not all
  // of the pointers below will be non-NULL.
  // Some of the online-feature pointers are just copies of other pointers,
  // and we do have to avoid deleting them in those cases.
  if (final_feature_ != feature_plus_optional_cmvn_)
    delete final_feature_;
  delete nvector_feature_;
  delete cmvn_feature_;
  if (feature_plus_optional_pitch_ != base_feature_)
    delete feature_plus_optional_pitch_;
  delete pitch_feature_;
  delete pitch_;
  delete base_feature_;
}

void OnlineNnet2NoiseFeaturePipeline::AcceptWaveform(
    BaseFloat sampling_rate,
    const VectorBase<BaseFloat> &waveform) {
  base_feature_->AcceptWaveform(sampling_rate, waveform);
  if (pitch_)
    pitch_->AcceptWaveform(sampling_rate, waveform);
}

void OnlineNnet2NoiseFeaturePipeline::InputFinished() {
  base_feature_->InputFinished();
  if (pitch_)
    pitch_->InputFinished();
}

BaseFloat OnlineNnet2NoiseFeaturePipelineInfo::FrameShiftInSeconds() const {
  if (feature_type == "mfcc") {
    return mfcc_opts.frame_opts.frame_shift_ms / 1000.0f;
  } else if (feature_type == "fbank") {
    return fbank_opts.frame_opts.frame_shift_ms / 1000.0f;
  } else if (feature_type == "plp") {
    return plp_opts.frame_opts.frame_shift_ms / 1000.0f;
  } else {
    KALDI_ERR << "Unknown feature type " << feature_type;
    return 0.0;
  }
}


}  // namespace kaldi
