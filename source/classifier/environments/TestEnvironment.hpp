#pragma once

#include "ExecEnvironment.hpp"
#include "../results/TestResult.hpp"

namespace spp {
    namespace envs {

        class TestEnvironment : public ExecEnvironment {
        private:
            std::vector<Data>& _files;

        public:
            explicit TestEnvironment(std::vector<Data>& files);

            TestResult* run(RCNN net) override;
        };

    }
}