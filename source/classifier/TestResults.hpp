#pragma once

#include <vector>
#include <string>
#include <torch/torch.h>
#include "data.hpp"

namespace spp {
    class TestResults {
    private:
        float avg_loss, l_min, l_max, sd, p_correct;
        int n_correct;
        int matrix[6][6] = {0};
        std::vector<float> losses;

        void calculate();

        float stddev();

        int updateMatrix(const torch::Tensor& output, int lang);

        void printMatrix();

    public:

        TestResults();

        void registerTest(float loss, const torch::Tensor& output, Language label);

        void print();

        std::string printCSV();
    };
}


