#include "TestResults.hpp"

namespace spp {
    int TestResults::updateMatrix(const torch::Tensor& output, int lang) {
        float vals[6];
        int classified = 0;
        for (int i = 0; i < 6; ++i) vals[i] = output.data_ptr<float>()[i];

        int i;
        for (i = 0; i < 6; ++i) {
            float m = vals[0];
            int max_i = 0;
            for (int j = 1; j < 6; ++j) {
                if (vals[j] > m) {
                    m = vals[j];
                    max_i = j;
                }
            }
            if (i == 0) classified = max_i;
            vals[max_i] = -1000.0;
            if (max_i == lang) break;
        }
        matrix[i][lang]++;
        return classified;
    }

    void TestResults::printMatrix() {
        std::cout << "  NL  |  EN  |  DE  |  FR  |  ES  |  IT  |\n"
                  << "------+------+------+------+------+------+\n";
        for (auto& i : matrix) {
            for (int j : i) {
                std::printf(" %4d |", j);
            }
            std::cout << std::endl;
        }
    }

    float TestResults::stddev() {
        float std = 0;
        float mean = 0;
        for (auto& v : losses) mean += v;
        mean /= losses.size();
        for (auto& v : losses) std += (v - mean) * (v - mean);
        return std::sqrt(std / losses.size());
    }


    TestResults::TestResults() : avg_loss(-1), l_min(1000.0), l_max(-1000.0), n_correct(0), sd(-1),
                                 p_correct(-1) {
    }

    void TestResults::registerTest(float loss, const torch::Tensor& output, Language label) {
        losses.push_back(loss);
        int classified = updateMatrix(output, label);
        if (label == classified) n_correct++;
        l_max = std::max(l_max, loss);
        l_min = std::min(l_min, loss);
    }

    void TestResults::calculate() {
        sd = stddev();
        avg_loss = 0;
        for (auto& v : losses) avg_loss += v;
        avg_loss /= losses.size();
        p_correct = static_cast<float>(n_correct) / losses.size();
    }

    void TestResults::print() {
        calculate();
        std::cout << "Average Loss: " << avg_loss << std::endl
                  << "Max Loss: " << l_max << ", Min Loss: " << l_min << std::endl
                  << "Loss Standard Deviation: " << sd << std::endl
                  << "Classification Accuracy: " << n_correct << "/" << losses.size() << " --> "
                  << p_correct << std::endl << std::endl;
        printMatrix();
        std::cout << std::endl << "______________________________________" << std::endl;
    }

    std::string TestResults::printCSV() {
        calculate();
        std::stringstream ss;
        ss << avg_loss << "," << l_max << "," << l_min << "," << sd << "," << n_correct << ","
           << losses.size() << "," << p_correct;
        return ss.str();
    }
}


