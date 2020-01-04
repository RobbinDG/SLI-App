#include "RCNN.hpp"
#include "libraries/openmp3/openmp3.h"
#include "test.hpp"

namespace spp {

    TestResults test(RCNN net, const std::vector<Data>& files) {
        float buffer[2][SAMPLE_SIZE];
        TestResults results;

        for (const auto& file : files) {
            mp3ToSample(file.data, buffer);
            torch::Tensor output;
            torch::Tensor loss;

            if (buffer[0][0] != -2) {
                auto input = torch::from_blob(buffer, {2, SAMPLE_SIZE}).unsqueeze(0);

                torch::NoGradGuard no_grad_guard;
                output = net->forward(input);
                loss = torch::nll_loss(output.squeeze(0), LABELS[file.language]);
                results.registerTest(loss.data_ptr<float>()[0], output, file.language);
            }
        }

        results.print();
        return results;
    }

    Language classify(RCNN net, const std::string& file, bool save) {
        float buffer[2][SAMPLE_SIZE];
        static float buffer2[2][SAMPLE_SIZE];
        torch::Tensor output;

        mp3ToSample(file, buffer);
        if (save) {
            for (int i = 0; i < SAMPLE_SIZE; ++i)
                if (buffer[0][i] != buffer2[0][i])
                    std::cout << i << ": " << buffer[0][i] << " " << buffer2[0][i] << std::endl;
            mp3ToSample(file, buffer2);
        }
        auto input = torch::from_blob(buffer, {2, SAMPLE_SIZE}).unsqueeze(0);

        torch::NoGradGuard no_grad_guard;
        output = net->forward(input);


        int max_i = -1;
        float m = -1000.0;
        for (int i = 0; i < 6; ++i) {
            if (output.data_ptr<float>()[i] > m) {
                m = output.data_ptr<float>()[i];
                max_i = i;
            }
        }
        return static_cast<Language>(max_i);
    }

    void dumpParameters(RCNN& net, TestResults& results, int epoch, int batch) {
        std::stringstream ss;
        ss << "../params/params_" << epoch << "-" << batch;
        torch::save(net, ss.str());
        std::ofstream fs(TRAIN_STATS_FILE, std::ios_base::app);
        fs << epoch << "," << batch << "," << results.printCSV() << std::endl;
        fs.close();
    }

}