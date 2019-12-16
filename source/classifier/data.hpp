#pragma once

#include <map>
#include "RCNN.hpp"
#include "libraries/openmp3/openmp3.h"

#define TRAIN_CYCLE 400
#define SAMPLE_SIZE 2304
#define FRAME_LENGTH 1152
#define COMPRESSION 5
#define EPOCH_START 40
#define EPOCH_LIMIT 40

//#define NUM_FILES 596
#define DATA_ROOT "../../../trainingdata/"

namespace spp {
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

    typedef struct Data {
        std::string data;
        Language language;
    } Data;

    typedef struct SampleList {
        char* samples;
        size_t length;
    } SampleList;

    std::vector<Data> trainingData(const std::string& dir, int N);

    std::vector<int> distinctRandomList(int start, int end, int m);

    void normalise(float arr[2][SAMPLE_SIZE]);

    void dumpParameters(RCNN net, int epoch, int batch);

    SampleList readFile(const std::string& file);

    bool getTrainData(OpenMP3::Iterator& it, OpenMP3::Decoder& decoder,
                      float buffer[2][SAMPLE_SIZE]);

}