#pragma once

#include "data.hpp"
#include "results/TestResult.hpp"

namespace spp {

    void dumpParameters(CNN& net, TestResult& results, int epoch, int batch);

}