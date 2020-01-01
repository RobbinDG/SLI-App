#include <cstdlib>
#include <dirent.h>
#include "data.hpp"
#include "train.hpp"
#include "test.hpp"

void errorUsage(char** argv) {
    std::cerr << "Usage: " << argv[0] << " <method> <parameters...>";
    exit(EXIT_FAILURE);
}

void parseAndExecute(RCNN& net, int argc, char** argv) {
    if (argc > 1) {
        switch (std::atoi(argv[1])) {
            case spp::K_FOLD_CROSS_VALIDATION:
                if (argc >= 5) {
                    for (int epoch = spp::EPOCH_START; epoch < spp::EPOCH_LIMIT; ++epoch) {
                        auto data = spp::trainingData(argv[4], std::atoi(argv[2]));
                        spp::k_fold_cross_validation(net, data,
                                                     std::atoi(argv[3]), epoch);
                    }
                } else {
                    errorUsage(argv);
                }
                break;
            case spp::TEST_DIRECTORY:
                if (argc >= 3) {
                    std::cout << "Testing files in \"" << argv[2] << "\"\n";
                    DIR* dirp = opendir(argv[2]);
                    std::vector<spp::Data> v;
                    struct dirent* dp;
                    std::ofstream stream("../filesread.txt");
                    while ((dp = readdir(dirp)) != nullptr) {
                        char* ss = nullptr;
                        ss = std::strstr(dp->d_name, ".mp3");
                        if (!ss) continue;
                        spp::Data d;
                        std::string str(dp->d_name);
                        stream << dp->d_name << std::endl;
                        d.data = argv[2] + str;
                        std::string lang(str.begin(), str.begin() + 2);
                        d.language =
                                (lang == "nl") ? spp::Language::NL :
                                (lang == "en") ? spp::Language::EN :
                                (lang == "de") ? spp::Language::DE :
                                (lang == "fr") ? spp::Language::FR :
                                (lang == "es") ? spp::Language::ES :
                                spp::Language::IT;
                        v.emplace_back(d);
                    }
                    stream.close();
                    closedir(dirp);
                    spp::test(net, v);
                } else {
                    errorUsage(argv);
                }
                break;
            case spp::TEST_FILE:
                if (argc >= 3) {
                    std::cout << "Testing \"" << argv[2] << "\"\n";
                    std::cout << spp::classify(net, argv[2]) << std::endl;
                } else {
                    errorUsage(argv);
                }
                break;
        }
    } else {
        errorUsage(argv);
    }
}

