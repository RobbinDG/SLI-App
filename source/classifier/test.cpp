#include "RCNN.hpp"
#include "libraries/openmp3/openmp3.h"
#include "test.hpp"

namespace spp {

    void updateMatrix(int matrix[6][6], const torch::Tensor& output, int lang) {
        float vals[6];
        for (int i = 0; i < 6; ++i) vals[i] = output.data_ptr<float>()[i];

        int i;
        for (i = 0; i < 6; ++i) {
            float max = vals[0];
            int max_i = 0;
            for (int j = 1; j < 6; ++j) {
                if (vals[j] > max) {
                    max = vals[j];
                    max_i = j;
                }
            }
            vals[max_i] = -1000.0;
            if (max_i == lang) break;
        }
        matrix[i][lang]++;
    }

    void printMatrix(int matrix[6][6]) {
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                std::cout << matrix[i][j] << "|";
            }
            std::cout << std::endl;
        }
    }

    float stddev(std::vector<float>& vals) {
        float std = 0;
        float mean = 0;
        for (auto& v : vals) mean += v;
        mean /= vals.size();
        for (auto& v : vals) std += (v - mean) * (v - mean);
        return std::sqrt(std / vals.size());
    }

    void test(RCNN net, const std::vector<Data>& files) {
        OpenMP3::Library openmp3;
        OpenMP3::Decoder decoder(openmp3);

        float buffer[2][SAMPLE_SIZE];
        float total_loss = 0.0;

        float t_max = -1.0, t_min = 1000;
        std::vector<float> losses;
        int classification_matrix[6][6] = {0};

        for (const auto & file : files) {
            SampleList sl = readFile(file.data);
            OpenMP3::Iterator it(openmp3, reinterpret_cast<const OpenMP3::UInt8*>(sl.samples),
                                 sl.length);
            int frame_idx = 0;
            torch::Tensor avg_loss = torch::zeros({1});
            torch::Tensor output;
            torch::Tensor avg_output = torch::zeros({1,1,6});

            while (getTrainData(it, decoder, buffer)) {
                auto input = torch::from_blob(buffer, {2, SAMPLE_SIZE}).unsqueeze(0);

                torch::NoGradGuard no_grad_guard;
                output = net->forward(input);
                auto loss = torch::nll_loss(output.squeeze(0), LABELS[file.language]);

                avg_loss += loss;
                avg_output += output;
                t_min = std::min(t_min, loss.data_ptr<float>()[0]);
                t_max = std::max(t_max, loss.data_ptr<float>()[0]);
                losses.push_back(loss.data_ptr<float>()[0]);
                frame_idx++;
            }

            torch::Tensor loss_true = avg_loss / frame_idx;
            torch::Tensor out_true = avg_output / frame_idx;
            updateMatrix(classification_matrix, out_true, file.language);

            total_loss += loss_true.data_ptr<float>()[0] / files.size();
            delete[] sl.samples;
        }
        int correct_classifications = 0;
        for (int i = 0; i < 6; ++i) correct_classifications += classification_matrix[0][i];

        std::cout << "Test Error: " << total_loss << std::endl
                  << "Test Max: " << t_max << ", Test Min: " << t_min << std::endl
                  << "Test Standard Deviation: " << stddev(losses) << std::endl
                  << "Classification Accuracy: "
                  << correct_classifications / files.size()
                  << std::endl << std::endl;
        printMatrix(classification_matrix);
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
        torch::Tensor avg_output = torch::zeros({1,1,6});

        while (getTrainData(it, decoder, buffer)) {
            auto input = torch::from_blob(buffer, {2, SAMPLE_SIZE}).unsqueeze(0);

            torch::NoGradGuard no_grad_guard;
            output = net->forward(input);
            avg_output += output;
            frame_idx++;
        }
        torch::Tensor out_true = avg_output / frame_idx;
        delete[] sl.samples;

        int max_i;
        float m;
        for (int i = 0; i < 6; ++i) {
            if (out_true.data_ptr<float>()[i] > m) {
                m = out_true.data_ptr<float>()[i];
                max_i = i;
            }
        }
        return static_cast<Language>(max_i);
    }

}