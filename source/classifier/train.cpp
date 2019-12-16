#include <random>
#include "RCNN.hpp"
#include "libraries/openmp3/openmp3.h"
#include "test.hpp"
#include "train.hpp"

namespace spp {

    void
    k_fold_cross_validation(RCNN& net, const std::vector<Data>& files, int K, int epoch) {
        size_t N = files.size() / 6;
        std::vector<std::vector<Data>> v(K);

        // Store files by language and shuffle
        std::vector<std::vector<Data>> buckets(6);
        for (int i = 0; i < 6; ++i) {
            buckets[i] = std::vector<Data>(files.begin() + i * N,
                                                  files.begin() + (i + 1) * N);
            std::shuffle(buckets[i].begin(), buckets[i].end(),
                         std::mt19937(std::random_device()()));
        }

        // Distribute files equally over the partitions
        int b = 0, k = 0, i = 0;
        for (size_t x = 0; x < files.size(); ++x) {
            v[k++].push_back(buckets[b][i++]);
            if (i >= buckets[b].size()) {
                b++;
                i = 0;
            }
            if (k >= v.size()) k = 0;
        }

        // Shuffle all partitions
        for (k = 0; k < K; ++k)
            std::shuffle(v[k].begin(), v[k].end(), std::mt19937(std::random_device()()));

        // Train while leaving each partition out, once
        for (k = 0; k < K; ++k) {
            std::cout << "Training, excluding batch " << (k + 1) << "/" << K << std::endl;
            train_once(net, v, k);
            dumpParameters(net, epoch, k);
        }
    }

    void train_once(RCNN& net, const std::vector<std::vector<Data>>& files, int test_idx) {
        std::vector<Data> v;
        for (size_t i = 0; i < files.size(); ++i) {
            if (i != test_idx) v.insert(v.end(), files[i].begin(), files[i].end());
        }
        train(net, v);
        test(net, files[test_idx]);
    }

    void train(RCNN& net, const std::vector<Data>& files) {
        torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(5e-6));

        OpenMP3::Library openmp3;
        OpenMP3::Decoder decoder(openmp3);

        float buffer[2][SAMPLE_SIZE];

        for (const auto& file : files) {
            SampleList sl = readFile(file.data);
            OpenMP3::Iterator it(openmp3, reinterpret_cast<const OpenMP3::UInt8*>(sl.samples),
                                 sl.length);

            while (getTrainData(it, decoder, buffer)) {
                if (buffer[0][0] != -1) {
                    auto input = torch::from_blob(buffer, {2, SAMPLE_SIZE}).unsqueeze(0);

                    net->zero_grad();
                    auto output = net->forward(input);
                    auto loss = torch::nll_loss(output.squeeze(0), LABELS[file.language]);
                    loss.backward();
                    optimizer.step();
                }
            }
            delete[] sl.samples;
        }
    }
/*
    void train(RCNN net) {
        torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(5e-6));

        OpenMP3::Library openmp3;
        OpenMP3::Decoder decoder(openmp3);

        float buffer[2][SAMPLE_SIZE];

        for (int epoch = EPOCH_START; epoch < EPOCH_LIMIT; ++epoch) {
            std::cout << "Training epoch " << epoch << std::endl;
            auto files = trainingData(DATA_ROOT + "trainingdata/selection/");

            for (size_t i = 0; i < TRAIN_FRACTION * files.size(); ++i) {
                SampleList sl = readFile(files[i].data);
                OpenMP3::Iterator it(openmp3, reinterpret_cast<const OpenMP3::UInt8*>(sl.samples),
                                     sl.length);

                int frame_idx = 0;
                torch::Tensor avg_loss = torch::zeros({1});

                while (getTrainData(it, decoder, buffer)) {
                    if (buffer[0][0] != -1) {
                        auto input = torch::from_blob(buffer, {2, SAMPLE_SIZE}).unsqueeze(0);

                        net->zero_grad();
                        auto output = net->forward(input);
                        auto loss = torch::nll_loss(output.squeeze(0), LABELS[files[i].language]);
                        loss.backward();
                        optimizer.step();

                        avg_loss += loss;
                        frame_idx++;
                    }
                }

                auto l = avg_loss / frame_idx;
                delete[] sl.samples;
            }

            test(net, files);
            std::stringstream ss;
            ss << "params_" << epoch;
//            dumpParameters(net, ss.str());
        }
    }

*/
}

