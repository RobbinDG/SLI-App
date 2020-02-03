#pragma once

#include "VoidResult.hpp"
#include "../data.hpp"

namespace spp {
    /**
     * Represents the result of a classification.
     */
    class ClassifyResult : public VoidResult {
    private:
        Language _language;

    public:
        explicit ClassifyResult(Language language);

        void print() override;

    };

}
