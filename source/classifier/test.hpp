#pragma once

#include "data.hpp"

namespace spp {

    void test(RCNN net, const std::vector<Data>& files);

    Language classify(RCNN net, const std::string& file);

}