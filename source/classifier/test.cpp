#include "CNN.hpp"
#include "test.hpp"

namespace spp {

    void dumpParameters(CNN& net, TestResult& results, int epoch, int batch) {
        std::stringstream ss;
        ss << "../params/params_" << epoch << "-" << batch;
        torch::save(net, ss.str());

        results.print();

        std::ofstream fs(TRAIN_STATS_FILE, std::ios_base::app);
        fs << epoch << "," << batch << "," << results.printCSV() << std::endl;
        fs.close();
    }

}