#pragma once

#include "ExecEnvironment.hpp"
#include "../results/TestResult.hpp"

namespace spp {
    namespace envs {

        /**
         * An environment for classifying languages on a larger set of files
         */
        class TestEnvironment : public ExecEnvironment {
        private:
            std::vector<Data> _files;

        public:
            explicit TestEnvironment(std::vector<Data>& files);

            TestResult* run(CNN net) override;
        };

    }
}