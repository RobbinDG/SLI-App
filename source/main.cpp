#include <iostream>
#include <random>
#include "data.hpp"
#include "train.hpp"
#include "test.hpp"
#include <sys/types.h>
#include <dirent.h>


int main(int argc, char** argv) {
    RCNN rcnn(1152, 2);

    std::ifstream fs(spp::save_loc);
    if (fs.good()) {
        torch::load(rcnn, spp::save_loc);
        std::cout << "Successfully loaded model from file" << std::endl;
    }

    if (argc == 2) {
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

        std::cout << v.size() << std::endl;

        spp::test(rcnn, v, true);
    } else {
        spp::train(rcnn);
    }

    return 0;
}