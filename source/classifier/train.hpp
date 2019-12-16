#pragma once

#include "RCNN.hpp"

namespace spp {

    /**
     * Trains [net] on files. Assumes files are sorted by language in the correct order.
     */
    void k_fold_cross_validation(RCNN& net, const std::vector<Data>& files, int K, int epoch);

    /**
     * Trains [net] on files (excluding partition [test_idx]) and trains on partition [test_idx].
     */
    void train_once(RCNN& net, const std::vector<std::vector<Data>>& files, int test_idx);

    /**
     * Executes the train loop on all files
     */
    void train(RCNN& net, const std::vector<Data>& files);

}
