#include "ClassifyEnvironment.hpp"

#include <utility>

namespace spp {
    namespace envs {
        ClassifyEnvironment::ClassifyEnvironment(std::string file) : _file(std::move(file)) {
        }

        ClassifyResult* ClassifyEnvironment::run(CNN net) {
            net->eval();

            float buffer[1][SAMPLE_SIZE];
            torch::Tensor output;

            mp3ToSample(_file, buffer);
            auto input = torch::from_blob(buffer, {1, SAMPLE_SIZE}).unsqueeze(0);

            torch::NoGradGuard no_grad_guard;
            output = torch::exp(net->forward(input));

            std::vector<float> outVector;
            outVector.reserve(6);

            for (int i = 0; i < 6; ++i) outVector.push_back(output.data_ptr<float>()[i]);

            return new ClassifyResult(outVector);
        }
    }
}




