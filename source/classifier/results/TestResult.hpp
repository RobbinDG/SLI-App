#pragma once

#include <vector>
#include <string>
#include <torch/torch.h>
#include "../data.hpp"
#include "VoidResult.hpp"

namespace spp {
    class TestResult : public VoidResult {
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

        TestResult();

        void registerTest(float loss, const torch::Tensor& output, Language label);

        void print() override;

        std::string printCSV();
    };
}


