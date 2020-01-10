#include "TestEnvironment.hpp"

namespace spp {
    namespace envs {

        TestEnvironment::TestEnvironment(std::vector<Data>& files) : _files(files) {
        }

        TestResult* TestEnvironment::run(RCNN net) {
            auto* result = new TestResult();

            for (const auto& file : _files) {
                float buffer[2][SAMPLE_SIZE];
                mp3ToSample(file.data, buffer);
                torch::Tensor output;
                torch::Tensor loss;

                if (buffer[0][0] != -2) {
                    auto input = torch::from_blob(buffer, {2, SAMPLE_SIZE}).unsqueeze(0);

                    torch::NoGradGuard no_grad_guard;
                    output = net->forward(input);
                    loss = torch::nll_loss(output.squeeze(0), LABELS[file.language]);
                    result->registerTest(loss.data_ptr<float>()[0], output, file.language);
                }
            }
            return result;
        }
    }
}


