#include "RCNN.hpp"
#include "libraries/openmp3/openmp3.h"
#include "test.hpp"

namespace spp {

    void train(RCNN net) {
        torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(5e-6));

        OpenMP3::Library openmp3;
        OpenMP3::Decoder decoder(openmp3);
        OpenMP3::Frame frame;

        float buffer[2][SAMPLE_SIZE];

        for (int epoch = EPOCH_START; epoch < EPOCH_LIMIT; ++epoch) {
            std::cout << "Training epoch " << epoch << std::endl;
            auto files = trainingData("../../trainingdata/selection/");

            for (size_t i = 0; i < TRAIN_FRACTION * files.size(); ++i) {
                if (i % TRAIN_CYCLE == 0) {
                }
                SampleList sl = readFile(files[i].data);
                OpenMP3::Iterator it(openmp3, reinterpret_cast<const OpenMP3::UInt8*>(sl.samples),
                                     sl.length);


//                std::cout << "Training on \"" << files[i].data << "\"\n";

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
//                std::cout << l.accessor<float, 1>()[0] << std::endl;

                delete[] sl.samples;
            }

            test(net, files);
            std::stringstream ss;
            ss << "params_" << epoch;
            dumpParameters(net, ss.str());
        }
    }

}

