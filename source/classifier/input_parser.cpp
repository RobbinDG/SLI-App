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

std::unique_ptr<spp::envs::ExecEnvironment> parse(int argc, char** argv) {
    using namespace spp::envs;
    if (argc > 1) {
        switch (std::atoi(argv[1])) {
            case spp::K_FOLD_CROSS_VALIDATION:
                if (argc >= 5) {
                    auto data = spp::trainingData(argv[4], std::atoi(argv[2]));
                    KFoldCrossValidationEnv env(data, std::atoi(argv[3]), 1e-4, 0);
                    return std::make_unique<KFoldCrossValidationEnv>(env);
                } else {
                    errorUsage(argv);
                }
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

                    TestEnvironment env(v);
                    return std::make_unique<TestEnvironment>(v);
                } else {
                    errorUsage(argv);
                }
            case spp::TEST_FILE:
                if (argc >= 3) {
                    std::cout << "Testing \"" << argv[2] << "\"\n";
                    ClassifyEnvironment env(argv[2]);
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

