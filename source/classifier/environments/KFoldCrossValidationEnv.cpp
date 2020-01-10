#include <random>
#include "KFoldCrossValidationEnv.hpp"
#include "../test.hpp"
#include "TestEnvironment.hpp"

namespace spp {
    namespace envs {

        KFoldCrossValidationEnv::KFoldCrossValidationEnv(
                const std::vector<Data>& files,
                int K,
                float learningRate,
                int startEpoch,
                int epochLimit) :
                TrainEnvironment(files, learningRate, startEpoch, epochLimit),
                _K(K) {}

        void KFoldCrossValidationEnv::runEpoch(RCNN net, int epoch) {
            size_t N = _files.size() / 6;
            std::vector<std::vector<Data>> v(_K);

            // Store files by language and shuffle
            std::vector<std::vector<Data>> buckets(6);
            for (int i = 0; i < 6; ++i) {
                buckets[i] = std::vector<Data>(_files.begin() + i * N,
                                               _files.begin() + (i + 1) * N);
                std::shuffle(buckets[i].begin(), buckets[i].end(),
                             std::mt19937(std::random_device()()));
            }

            // Distribute files equally over the partitions
            int b = 0, k = 0, i = 0;
            for (size_t x = 0; x < _files.size(); ++x) {
                v[k++].push_back(buckets[b][i++]);
                if (i >= buckets[b].size()) {
                    b++;
                    i = 0;
                }
                if (k >= v.size()) k = 0;
            }

            // Shuffle all partitions
            for (k = 0; k < _K; ++k)
                std::shuffle(v[k].begin(), v[k].end(), std::mt19937(std::random_device()()));

            // Train while leaving each partition out, once
            for (k = 0; k < _K; ++k) {
                std::cout << "Training, excluding batch " << (k + 1) << "/" << _K << std::endl;
                auto test_results = train_once(net, v, k);
                dumpParameters(net, *test_results, epoch, k);
                delete test_results;
            }
        }

        TestResult*
        KFoldCrossValidationEnv::train_once(RCNN& net, const std::vector<std::vector<Data>>& files, int test_idx) {
            std::vector<Data> v;
            for (size_t i = 0; i < files.size(); ++i) {
                if (i != test_idx) v.insert(v.end(), files[i].begin(), files[i].end());
            }
            train(net, v);
            auto s = files[test_idx];
            TestEnvironment tenv(s);
            return tenv.run(net);
        }

        void KFoldCrossValidationEnv::train(RCNN& net, const std::vector<Data>& files) {
            torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(_learning_rate));

            float buffer[1][SAMPLE_SIZE];

            for (const auto& file : files) {
                mp3ToSample(file.data, buffer);
                if (buffer[0][0] != -2) {
                    auto input = torch::from_blob(buffer, {1, SAMPLE_SIZE}).unsqueeze(0);

                    net->zero_grad();
                    auto output = net->forward(input);
                    auto loss = torch::nll_loss(output.squeeze(0), LABELS[file.language]);
                    loss.backward();
                    optimizer.step();
                }
            }
        }
    }
}