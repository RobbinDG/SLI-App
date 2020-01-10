#pragma once

#include <map>
#include "RCNN.hpp"
#include "libraries/openmp3/openmp3.h"

namespace spp {
    extern const int EPOCH_START, EPOCH_LIMIT;
    constexpr int FRAME_LENGTH = 1152, COMPRESSION = 5, FRAMES_PER_SAMPLE = 96,
            SAMPLE_SIZE = FRAMES_PER_SAMPLE * FRAME_LENGTH / COMPRESSION;
    extern const std::string TRAIN_STATS_FILE;

    extern long _NL[6];
    extern long _EN[6];
    extern long _DE[6];
    extern long _FR[6];
    extern long _ES[6];
    extern long _IT[6];

    enum Language {
        NL, EN, DE, FR, ES, IT
    };

    extern torch::TensorOptions options;
    extern std::map<Language, const torch::Tensor> LABELS;

    extern std::vector<std::string> langs;
    extern std::string save_loc;

    enum Modes {
        K_FOLD_CROSS_VALIDATION = 0, // Requires: samples/language (N), groups (K), directory
        TEST_DIRECTORY,              // Requires: directory
        TEST_FILE                    // Requires: file
    };

    typedef struct {
        std::string data;
        Language language;
    } Data;

    typedef struct SampleList {
        char* samples;
        size_t length;
    } SampleList;

    std::vector<Data> trainingData(const std::string& dir, int N);

    std::vector<int> distinctRandomList(int start, int end, int m);

    void normalise(float buffer[2][SAMPLE_SIZE]);

    void medianFilter(float arr[2][SAMPLE_SIZE], int filter_size);

    float** stereoToMono(float arr[2][SAMPLE_SIZE]);

    SampleList readFile(const std::string& file);

    bool getTrainData(OpenMP3::Iterator& it, OpenMP3::Decoder& decoder,
                      float buffer[2][SAMPLE_SIZE]);

    void
    mp3ToSample(const std::string& file, float buffer[2][SAMPLE_SIZE]);
}