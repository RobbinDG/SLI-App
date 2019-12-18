#include <iostream>
#include <random>
#include "data.hpp"
#include "input_parser.hpp"

int main(int argc, char** argv) {
    RCNN rcnn(1152, 2);

    std::ifstream fs(spp::save_loc);
    if (fs.good()) {
        torch::load(rcnn, spp::save_loc);
        std::cout << "Successfully loaded model from file" << std::endl;
    }

    parseAndExecute(rcnn, argc, argv);

    return 0;
}