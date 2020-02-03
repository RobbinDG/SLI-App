#pragma once

#include "ExecEnvironment.hpp"
#include "../data.hpp"
#include "../results/ClassifyResult.hpp"

namespace spp {
    namespace envs {
        /**
         * An environment for classifying languages
         */
        class ClassifyEnvironment : public ExecEnvironment {
        private:
            std::string _file;

        public:
            explicit ClassifyEnvironment(std::string file);

            ClassifyResult* run(CNN net) override;
        };

    }
}