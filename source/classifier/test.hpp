#pragma once

#include "data.hpp"
#include "TestResults.hpp"

namespace spp {

    TestResults test(RCNN net, const std::vector<Data>& files);

    Language classify(RCNN net, const std::string& file, bool save = false);

    void dumpParameters(RCNN& net, TestResults& results, int epoch, int batch);

}