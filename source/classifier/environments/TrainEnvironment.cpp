#include "TrainEnvironment.hpp"

#include <utility>

namespace spp {
    namespace envs {
        TrainEnvironment::TrainEnvironment(std::vector<Data> files, float learning_rate,
                                           int start_epoch, int epoch_limit)
                : _files(std::move(files)),
                  _learning_rate(learning_rate),
                  _start_epoch(start_epoch),
                  _epoch_limit(epoch_limit) {}

        VoidResult* TrainEnvironment::run(RCNN net) {
            for (int e = _start_epoch; e < _epoch_limit; ++e)
                runEpoch(net, e);

            return new VoidResult();
        }
    }
}




