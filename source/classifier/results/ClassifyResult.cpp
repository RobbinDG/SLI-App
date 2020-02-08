#include "ClassifyResult.hpp"

namespace spp {
    void ClassifyResult::print() {
        for(auto& v : _probabilities) std::cout << v << " ";
        std::cout << std::endl;
    }

    ClassifyResult::ClassifyResult(std::vector<float>& output) : _probabilities(output) {
    }
}


