#pragma once

#include <map>
#include "CNN.hpp"
#include "libraries/openmp3/openmp3.h"

namespace spp {
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

    /**
     * Gathers Data objects from a given file directory.
     * Assumes equal amounts of files for each language
     * @param dir
     * @param N
     */
    std::vector<Data> trainingData(const std::string& dir, int N);

    /**
     * Normalises a frame.
     */
    void normalise(float buffer[2][SAMPLE_SIZE]);

    /**
     * Converts a dual channel mono frame to single channel
     */
    void compressMono(float arr[2][FRAME_LENGTH], float arr2[1][FRAME_LENGTH]);

    /**
     * Converts stereo frames to signle channel mono.
     */
    void stereoToMono(float arr[2][FRAME_LENGTH], float arr2[1][FRAME_LENGTH]);

    /**
     * Reads a file and returns the samples.
     */
    SampleList readFile(const std::string& file);

    /**
     * Converts an mp3 to samples and stores it into a buffer. Does necessary pre-processing
     * like normalisation and removing dead frames
     */
    void mp3ToSample(const std::string& file, float buffer[2][SAMPLE_SIZE]);
}