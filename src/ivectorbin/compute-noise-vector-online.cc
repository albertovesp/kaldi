// ivectorbin/compute-noise-vector-online.cc

// Copyright   2020   Johns Hopkins University (author: Desh Raj)

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
#include "matrix/kaldi-matrix.h"
#include "feat/feature-functions.h"
#include "ivector/online-noise-vector.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Compute online noise vectors for each utterance, using\n"
        "the prior parameters provided. The number of extracted\n"
        "vectors for each utterance depends on the length and the\n"
        "value of the _period_ parameter, which is similar to the\n"
        "ivector-period used in online i-vector estimation.\n"
        "Usage: compute-noise-vector [options] <feats-rspecifier> "
        " <targets-rspecifier> <noise-prior> <period> <matrix-wspecifier>\n"
        "E.g.: compute-noise-vector [options] scp:feats.scp scp:targets.scp 10 ark:-\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string feat_rspecifier, target_rspecifier, noise_prior_rxfilename, 
      matrix_wspecifier;
    int32 period;
    feat_rspecifier = po.GetArg(1),
      target_rspecifier = po.GetArg(2),
      noise_prior_rxfilename = po.GetArg(3),
      matrix_wspecifier = po.GetArg(5);
    period = std::stoi(po.GetArg(4));

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatMatrixWriter matrix_writer(matrix_wspecifier);
    RandomAccessBaseFloatMatrixReader target_reader(target_rspecifier);
    OnlineNoisePrior noise_prior;
    ReadKaldiObject(noise_prior_rxfilename, &noise_prior);

    int32 num_done = 0, num_err = 0;
    OnlineNoiseVector noise_vec(noise_prior, period);

    for (;!feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      const Matrix<BaseFloat> &feat = feat_reader.Value();
      if (feat.NumRows() == 0) {
        KALDI_WARN << "Empty feature matrix for utterance " << utt;
        num_err++;
        continue;
      }
      Matrix<BaseFloat> noise_vectors;

      if (!target_reader.HasKey(utt)) {
        KALDI_WARN << "No target found for utterance. Getting noise vector "
          "from prior estimate." << utt;
        num_err++;
        noise_vec.ExtractVectors(feat, &noise_vectors);
      } else {
        const Matrix<BaseFloat> &target = target_reader.Value(utt);
        std::vector<bool> silence_decisions;

        if (feat.NumRows() != target.NumRows()) {
          KALDI_WARN << "Mismatch in number for frames " << feat.NumRows()
                     << " for features and targets " << target.NumRows()
                     << ", for utterance " << utt
                     << ". Creating vector from prior estimate.";
          num_err++;
          noise_vec.ExtractVectors(feat, &noise_vectors);
        } else {
          for (int32 i = 0; i < feat.NumRows(); i++) {
            silence_decisions.push_back(target(i,0) > target(i,1) || 
                target(i,2) > target(i,1));
          }
          noise_vec.ExtractVectors(feat, silence_decisions, &noise_vectors);
        }
      }
      matrix_writer.Write(utt, noise_vectors);
      num_done++;
    }

    KALDI_LOG << "Done computing average noise frames; processed "
              << num_done << " utterances, "
              << num_err << " had errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
