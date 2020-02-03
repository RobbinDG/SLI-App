#include <cstdlib>
#include <dirent.h>
#include "data.hpp"
#include "test.hpp"
#include "input_parser.hpp"
#include "environments/KFoldCrossValidationEnv.hpp"
#include "environments/TestEnvironment.hpp"
#include "environments/ClassifyEnvironment.hpp"

void errorUsage(char** argv) {
    std::cerr << "Usage: " << argv[0] << " <method> <parameters...>";
    exit(EXIT_FAILURE);
}

/**
 * Parses the input arguments
 * @return a smart pointer to a set up execution environment, nullptr in case of failure.
 */
std::unique_ptr<spp::envs::ExecEnvironment> parse(int argc, char** argv) {
    using namespace spp::envs;
    if (argc > 0) {
        switch (std::atoi(argv[0])) {
            case spp::K_FOLD_CROSS_VALIDATION:
                if (argc >= 4) {
                    auto data = spp::trainingData(argv[3], std::atoi(argv[1]));
                    KFoldCrossValidationEnv env(data, std::atoi(argv[2]), 3e-5, 4);
                    return std::make_unique<KFoldCrossValidationEnv>(env);
                } else {
                    errorUsage(argv);
                }
            case spp::TEST_DIRECTORY:
                if (argc >= 2) {
                    std::cout << "Testing files in \"" << argv[1] << "\"\n";
                    DIR* dirp = opendir(argv[1]);
                    std::vector<spp::Data> v;
                    struct dirent* dp;
                    while ((dp = readdir(dirp)) != nullptr) {
                        char* ss = nullptr;
                        ss = std::strstr(dp->d_name, ".mp3");
                        if (!ss) continue;
                        spp::Data d;
                        std::string str(dp->d_name);
                        d.data = argv[1] + str;
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
                    closedir(dirp);

                    TestEnvironment env(v);
                    return std::make_unique<TestEnvironment>(v);
                } else {
                    errorUsage(argv);
                }
            case spp::TEST_FILE:
                if (argc >= 2) {
                    std::cout << "Testing \"" << argv[1] << "\"\n";
                    ClassifyEnvironment env(argv[1]);
                    return std::make_unique<ClassifyEnvironment>(env);
                } else {
                    errorUsage(argv);
                }
        }
    } else {
        errorUsage(argv);
    }
    return nullptr;
}

