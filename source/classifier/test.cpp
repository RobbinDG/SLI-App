#include "RCNN.hpp"
#include "libraries/openmp3/openmp3.h"
#include "test.hpp"

namespace spp {

    TestResults test(RCNN net, const std::vector<Data>& files) {
        OpenMP3::Library openmp3;
        OpenMP3::Decoder decoder(openmp3);

        float buffer[2][SAMPLE_SIZE];
        TestResults results;

        for (const auto& file : files) {
            SampleList sl = readFile(file.data);
            OpenMP3::Iterator it(openmp3, reinterpret_cast<const OpenMP3::UInt8*>(sl.samples),
                                 sl.length);
            int frame_idx = 0;
            torch::Tensor output;
            torch::Tensor avg_loss = torch::zeros({1});
            torch::Tensor avg_output = torch::zeros({1, 1, 6});

            while (getTrainData(it, decoder, buffer)) {
                auto input = torch::from_blob(buffer, {2, SAMPLE_SIZE}).unsqueeze(0);

                torch::NoGradGuard no_grad_guard;
                output = net->forward(input);
                auto loss = torch::nll_loss(output.squeeze(0), LABELS[file.language]);

                avg_loss += loss;
                avg_output += output;
                frame_idx++;
            }

            results.registerTest(avg_loss.data_ptr<float>()[0] / frame_idx, avg_output / frame_idx,
                                 file.language);
            delete[] sl.samples;
        }

        results.print();
        return results;
    }

    Language classify(RCNN net, const std::string& file) {
        SampleList sl = readFile(file);
        float buffer[2][SAMPLE_SIZE];

        OpenMP3::Library openmp3;
        OpenMP3::Decoder decoder(openmp3);
        OpenMP3::Iterator it(openmp3, reinterpret_cast<const OpenMP3::UInt8*>(sl.samples),
                             sl.length);
        int frame_idx = 0;
        torch::Tensor output;
        torch::Tensor avg_output = torch::zeros({1, 1, 6});

        while (getTrainData(it, decoder, buffer)) {
            auto input = torch::from_blob(buffer, {2, SAMPLE_SIZE}).unsqueeze(0);

            torch::NoGradGuard no_grad_guard;
            output = net->forward(input);
            avg_output += output;
            frame_idx++;
        }
        torch::Tensor out_true = avg_output / frame_idx;
        delete[] sl.samples;

        int max_i = -1;
        float m = -1000.0;
        for (int i = 0; i < 6; ++i) {
            if (out_true.data_ptr<float>()[i] > m) {
                m = out_true.data_ptr<float>()[i];
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