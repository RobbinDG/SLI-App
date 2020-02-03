#pragma once

#include <vector>
#include <string>
#include <torch/torch.h>
#include "../data.hpp"
#include "VoidResult.hpp"

namespace spp {
    /**
     * Represents the result of a test phase.
     */
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

        /**
         * Registers the test of a single file, and immediately updates the statistics
         * @param loss
         * @param output
         * @param label
         */
        void registerTest(float loss, const torch::Tensor& output, Language label);

        void print() override;

        /**
         * Prints the result as a CSV row.
         */
        std::string printCSV();
    };
}


