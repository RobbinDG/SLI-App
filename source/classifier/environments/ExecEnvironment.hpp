#pragma once

#include "../RCNN.hpp"
#include "../results/VoidResult.hpp"

namespace spp { namespace envs {

    class ExecEnvironment {
    public:
        virtual VoidResult* run(RCNN net) = 0;

    };

} }


