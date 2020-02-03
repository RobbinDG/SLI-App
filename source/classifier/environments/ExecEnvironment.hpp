#pragma once

#include "../CNN.hpp"
#include "../results/VoidResult.hpp"

namespace spp {
    namespace envs {

        /**
         * An abstract environment to be overridden for ML
         * related tasks.
         */
        class ExecEnvironment {
        public:
            /**
             * Runs the environment on a given network
             * @param net
             * @return An abstract results object that contains the result
             * of the operation
             */
            virtual VoidResult* run(CNN net) = 0;

        };

    }
}


