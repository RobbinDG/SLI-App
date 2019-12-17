#include <vector>
#include <random>
#include "data.hpp"
#include "test.hpp"


namespace spp {
    long _NL[6] = {0};//{1, 0, 0, 0, 0, 0};
    long _EN[6] = {1};//{0, 1, 0, 0, 0, 0};
    long _DE[6] = {2};//{0, 0, 1, 0, 0, 0};
    long _FR[6] = {3};//{0, 0, 0, 1, 0, 0};
    long _ES[6] = {4};//{0, 0, 0, 0, 1, 0};
    long _IT[6] = {5};//{0, 0, 0, 0, 0, 1};

    torch::TensorOptions options = torch::TensorOptions(torch::kInt64);
    std::map<Language, const torch::Tensor> LABELS({
                                                           {NL, torch::from_blob(_NL, {1},
                                                                                 options)},
                                                           {EN, torch::from_blob(_EN, {1},
                                                                                 options)},
                                                           {DE, torch::from_blob(_DE, {1},
                                                                                 options)},
                                                           {FR, torch::from_blob(_FR, {1},
                                                                                 options)},
                                                           {ES, torch::from_blob(_ES, {1},
                                                                                 options)},
                                                           {IT, torch::from_blob(_IT, {1},
                                                                                 options)},
                                                   });

    std::vector<std::string> langs = {"nl", "en", "de", "fr", "es", "it"};
    std::string save_loc = "../params/serialised.pt";
/*
    int _gen() {
        static int i = 0;
        i = (i % NUM_FILES);
        return i++;
    }

    std::vector<int> distinctRandomList(int start, int end, int m) {
        std::vector<int> out(end - start);
        std::generate(out.begin(), out.end(), [start]() { return start + _gen(); });
        std::shuffle(out.begin(), out.end(), std::mt19937(std::random_device()()));
        return std::vector<int>(out.begin(), out.begin() + m);
    }
*/
    std::vector<Data> trainingData(const std::string& dir, int N) {
        std::vector<Data> files;
        int langIdx = 0;

        for (const auto& lang : langs) {
            auto prefix = dir + lang + '_';
            for (int i = 0; i < N; ++i) {
                auto file = prefix + std::to_string(i) + ".mp3";
                Data d;
                d.data = file;
                d.language =
                        (lang == "nl") ? Language::NL :
                        (lang == "en") ? Language::EN :
                        (lang == "de") ? Language::DE :
                        (lang == "fr") ? Language::FR :
                        (lang == "es") ? Language::ES :
                        Language::IT;
                files.push_back(d);
            }
            langIdx++;
        }
        return files;
    }

    void normalise(float arr[2][SAMPLE_SIZE]) {
        float maxs[2] = {-1000.0, -1000.0}, mins[2] = {1000.0, 1000.0};
        for (int i = 0; i < SAMPLE_SIZE; ++i) {
            for (int x = 0; x < 2; ++x) {
                maxs[x] = std::max(maxs[x], arr[x][i]);
                mins[x] = std::min(mins[x], arr[x][i]);
            }
        }

        if (maxs[0] == mins[0] || maxs[1] == mins[1]) {
            // useless frame, probably emtpy. Signal this to calling function
            arr[0][0] = -1;
        } else {
            float divs[2] = {std::max(maxs[0], -mins[0]), std::max(maxs[1], -mins[1])};
            for (int i = 0; i < SAMPLE_SIZE; ++i) {
//                for (int x = 0; x < 2; ++x) arr[x][i] = (arr[x][i] - mins[x]) / (maxs[x] - mins[x]);
                for (int x = 0; x < 2; ++x) arr[x][i] /= divs[x];
            }
        }
    }

    SampleList readFile(const std::string& file) {
        std::ifstream fl(file);
        fl.seekg(0, std::ios::end);
        size_t len = fl.tellg();
        char* ret = new char[len];
        fl.seekg(0, std::ios::beg);
        fl.read(ret, len);
        fl.close();
        return {ret, len};
    }

    bool getTrainData(OpenMP3::Iterator& it, OpenMP3::Decoder& decoder,
                      float buffer[2][SAMPLE_SIZE]) {
        OpenMP3::Frame frame;
        float t_buffer[COMPRESSION * (SAMPLE_SIZE / FRAME_LENGTH)][2][FRAME_LENGTH];

        bool hasFrames = true;
        for (auto& b : t_buffer) {
            if (it.GetNext(frame)) {
                decoder.ProcessFrame(frame, b);
            } else {
                for (int i = 0; i < FRAME_LENGTH; ++i) {
                    for (int x = 0; x < 1; ++x) b[x][i] = 0;
                }
                hasFrames = false;
            }
        }
        for (int i = 0; i < SAMPLE_SIZE; ++i) {
            int s = i * COMPRESSION;
            buffer[0][i] = t_buffer[s / FRAME_LENGTH][0][s % FRAME_LENGTH];
            buffer[1][i] = t_buffer[s / FRAME_LENGTH][1][s % FRAME_LENGTH];
        }
        normalise(buffer);
        return hasFrames;
    }

}