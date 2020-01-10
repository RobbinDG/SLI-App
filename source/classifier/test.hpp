#pragma once

#include "data.hpp"
#include "results/TestResult.hpp"

namespace spp {

    void dumpParameters(RCNN& net, TestResult& results, int epoch, int batch);

}