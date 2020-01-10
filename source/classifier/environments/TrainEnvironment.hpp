#pragma once

#include <climits>

#include "ExecEnvironment.hpp"
#include "../data.hpp"

namespace spp {
    namespace envs {

        class TrainEnvironment : public ExecEnvironment {
        private:
            int _start_epoch, _epoch_limit;

        protected:
            float _learning_rate;
            std::vector<Data> _files;

        public:
            TrainEnvironment(std::vector<Data> files, float learning_rate, int start_epoch = 0,
                             int epoch_limit = INT_MAX);

            VoidResult* run(RCNN net) override;

            virtual void runEpoch(RCNN net, int epoch) = 0;
        };

    }
}


