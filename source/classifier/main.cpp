#include <iostream>
#include <random>
#include "data.hpp"
#include "input_parser.hpp"

int main(int argc, char** argv) {
    CNN cnn;

    if (argc >= 2) {
        std::ifstream fs(argv[1]);
        if (fs.good()) {
            torch::load(cnn, argv[1]);
            cnn->eval();
            std::cout << "Successfully loaded model from file" << std::endl;
        }
    }

    auto env = parse(argc - 2, argv + 2);
    auto result = env->run(cnn);
    result->print();
    delete result;

    return 0;
}