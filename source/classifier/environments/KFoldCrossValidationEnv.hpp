#pragma once

#include "TrainEnvironment.hpp"
#include "../results/TestResult.hpp"

namespace spp {
    namespace envs {

        class KFoldCrossValidationEnv : public TrainEnvironment {
        private:
            int _K;

            /**
             * Trains [net] on files (excluding partition [test_idx]) and trains on partition [test_idx].
             */
            TestResult* train_once(RCNN& net, const std::vector<std::vector<Data>>& files, int test_idx);

            /**
             * Executes the train loop on all files
             */
            void train(RCNN& net, const std::vector<Data>& files);

        public:
            KFoldCrossValidationEnv(const std::vector<Data>& files, int K, float learningRate,
                                    int startEpoch = 0, int epochLimit = INT_MAX);
            /**
             * Trains [net] on files. Assumes files are sorted by language in the correct order.
             */
            void runEpoch(RCNN net, int epoch) override;
        };

    }
}


