#pragma once

#include "ExecEnvironment.hpp"
#include "../data.hpp"
#include "../results/ClassifyResult.hpp"

namespace spp {
    namespace envs {

        class ClassifyEnvironment : public ExecEnvironment {
        private:
            std::string _file;

        public:
            explicit ClassifyEnvironment(std::string file);

            ClassifyResult* run(RCNN net) override;
        };

    }
}