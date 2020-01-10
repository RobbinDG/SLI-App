#include "ClassifyResult.hpp"

namespace spp {
    void ClassifyResult::print() {
        std::cout << _language << std::endl;
    }

    ClassifyResult::ClassifyResult(Language language) : _language(language) {
    }
}


