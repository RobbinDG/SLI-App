#include <vector>
#include <random>
#include "data.hpp"

namespace spp {
    const std::string TRAIN_STATS_FILE = "../params/train_stats.csv";

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

    void normalise(float arr[1][SAMPLE_SIZE]) {
        float max = -1000.0, min = 1000.0;
        for (size_t i = 0; i < SAMPLE_SIZE; ++i) {
            max = std::max(max, arr[0][i]);
            min = std::min(min, arr[0][i]);
        }

        if (max == min) {
            // useless frame, probably emtpy. Signal this to calling function
            arr[0][0] = -2;
        } else {
            float div = std::max(max, -min);
            for (size_t i = 0; i < SAMPLE_SIZE; ++i) {
                arr[0][i] /= div;
            }
        }
    }

    void compressMono(float arr[2][FRAME_LENGTH], float arr2[1][FRAME_LENGTH]) {
        for (int i = 0; i < FRAME_LENGTH; ++i) arr2[0][i] = arr[0][i];
    }

    void stereoToMono(float arr[2][FRAME_LENGTH], float arr2[1][FRAME_LENGTH]) {
        for (int i = 0; i < FRAME_LENGTH; ++i) arr2[0][i] = (arr[0][i] + arr[1][i]) / 2;
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

    void
    mp3ToSample(const std::string& file, float buffer[1][SAMPLE_SIZE]) {
        OpenMP3::Library openmp3;
        OpenMP3::Decoder dec(openmp3);

        SampleList sl = readFile(file);
        OpenMP3::Iterator it(openmp3, reinterpret_cast<const OpenMP3::UInt8*>(sl.samples),
                             sl.length);
        OpenMP3::Frame frame;

        int q = 0;
        float t_buffer[FRAMES_PER_SAMPLE][1][FRAME_LENGTH];
        bool framesLeft = true;

        for (auto& b : t_buffer) {
            if (it.GetNext(frame)) {
                float stereo[2][FRAME_LENGTH];
                dec.ProcessFrame(frame, stereo);
                if (frame.GetMode() == OpenMP3::kModeMono)
                    compressMono(stereo, b);
                else
                    stereoToMono(stereo, b);
                q++;
            } else {
                framesLeft = false;
                break;
            }
        }

        // consume remaining frames
        if (framesLeft) {
            while (it.GetNext(frame));
        }

        for (int i = 0; i < SAMPLE_SIZE; ++i) {
            int s = i * COMPRESSION;
            buffer[0][i] = t_buffer[(s / FRAME_LENGTH) % q][0][s % FRAME_LENGTH];
        }
        normalise(buffer);
        delete[] sl.samples;
    }

}