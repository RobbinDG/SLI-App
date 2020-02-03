#pragma once

#include <climits>

#include "ExecEnvironment.hpp"
#include "../data.hpp"

namespace spp {
    namespace envs {

        /**
         * An abstract training environment for implementing different methods and strategies.
         */
        class TrainEnvironment : public ExecEnvironment {
        private:
            int _start_epoch, _epoch_limit;

        protected:
            float _learning_rate;
            std::vector<Data> _files;

        public:
            /**
             * Constructor
             * @param files the files to train the network on
             * @param learning_rate
             * @param start_epoch the epoch from which to start training, default = 0
             * @param epoch_limit the epoch upper bound, default = INF (continues until interrupted)
             */
            TrainEnvironment(std::vector<Data> files, float learning_rate, int start_epoch = 0,
                             int epoch_limit = INT_MAX);

            VoidResult* run(CNN net) override;

            /**
             * Runs an epoch in the training cycle.
             * @param net
             * @param epoch
             */
            virtual void runEpoch(CNN net, int epoch) = 0;
        };

    }
}


