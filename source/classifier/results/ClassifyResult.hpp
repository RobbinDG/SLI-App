#pragma once

#include "VoidResult.hpp"
#include "../data.hpp"

namespace spp {
    /**
     * Represents the result of a classification.
     */
    class ClassifyResult : public VoidResult {
    private:
        std::vector<float> _probabilities;

    public:
        explicit ClassifyResult(std::vector<float>& output);

        void print() override;

    };

}
