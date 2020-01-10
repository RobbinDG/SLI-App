#include "ClassifyEnvironment.hpp"

#include <utility>

namespace spp {
    namespace envs {
        ClassifyEnvironment::ClassifyEnvironment(std::string file) : _file(std::move(file)) {
        }

        ClassifyResult* ClassifyEnvironment::run(RCNN net) {
            float buffer[1][SAMPLE_SIZE];
            torch::Tensor output;

            mp3ToSample(_file, buffer);
            auto input = torch::from_blob(buffer, {1, SAMPLE_SIZE}).unsqueeze(0);

            torch::NoGradGuard no_grad_guard;
            output = net->forward(input);


            int max_i = -1;
            float m = -1000.0;
            for (int i = 0; i < 6; ++i) {
                if (output.data_ptr<float>()[i] > m) {
                    m = output.data_ptr<float>()[i];
                    max_i = i;
                }
            }
            return new ClassifyResult(static_cast<Language>(max_i));
        }
    }
}




