#include <iostream>
#include <random>
#include "data.hpp"
#include "input_parser.hpp"

int main(int argc, char** argv) {
    RCNN rcnn;

    if (argc >= 2) {
        std::ifstream fs(argv[1]);
        if (fs.good()) {
            torch::load(rcnn, spp::save_loc);
            std::cout << "Successfully loaded model from file" << std::endl;
        }
    }

    auto env = parse(argc - 2, argv + 2);
    auto result = env->run(rcnn);
    result->print();
    delete result;

    return 0;
}