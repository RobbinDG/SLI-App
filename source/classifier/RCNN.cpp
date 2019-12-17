#include "RCNN.hpp"

/*
RCNNImpl::RCNNImpl(int input_size, int channels) :
        features(register_module("features", torch::nn::Sequential(
                // Pre-processing
                // 2 Channels using 32 kernels or size 9 -> 32 channels x 128 samples
                torch::nn::Conv1d(
                        torch::nn::Conv1dOptions(2, SCALE * 16, 9)
                                .stride(9)
                ),
                torch::nn::Functional(torch::relu),
                torch::nn::BatchNorm(SCALE * 16),
                // 16 Channels using 32 kernels or size 3 -> 32 x 126
                torch::nn::Conv1d(torch::nn::Conv1dOptions(SCALE * 16, SCALE * 16, 3)),
                torch::nn::Functional(torch::relu),
                torch::nn::BatchNorm(SCALE * 16),
                torch::nn::MaxPool1d(3), // -> 32 x 42
                torch::nn::Conv1d(torch::nn::Conv1dOptions(SCALE * 16, SCALE * 32, 5)), // -> 64 x 38
                torch::nn::Functional(torch::relu),
                torch::nn::BatchNorm(SCALE * 32),
                torch::nn::MaxPool1d(38) // -> 32 x 1
        ))),
        dense(register_module("dense", torch::nn::Sequential(
                torch::nn::Linear(SCALE * 32 + 6, 6),
                torch::nn::Dropout(0.1)
        ))),
        outputFunction(register_module("outputFunction", torch::nn::Functional(torch::relu))) {}

torch::Tensor RCNNImpl::forward(const torch::Tensor& input, const torch::Tensor& prev) {
    auto t = features->forward(input);
    t = t.transpose(2, 1);
    t = torch::cat({t, prev}, 2);
    t = dense->forward(t);
    return torch::log_softmax(t, 2);
}

RCNNImpl::RCNNImpl(int input_size, int channels) :
        features(register_module("features", torch::nn::Sequential(
                // Pre-processing
                // 2 Channels using 32 kernels or size 9 -> 32 channels x 96 samples
                torch::nn::Conv1d(
                        torch::nn::Conv1dOptions(2, SCALE * 16, 12)
                                .stride(12)
                ),
                torch::nn::Functional(torch::relu),
                torch::nn::BatchNorm(SCALE * 16),
                // 16 Channels using 32 kernels or size 3 -> 64 x 47
                torch::nn::Conv1d(torch::nn::Conv1dOptions(SCALE * 16, SCALE * 32, 4).stride(2)),
                torch::nn::Functional(torch::relu),
                torch::nn::BatchNorm(SCALE * 32),
                torch::nn::MaxPool1d(3), // -> 64 x 15
                torch::nn::Conv1d(torch::nn::Conv1dOptions(SCALE * 32, SCALE * 64, 3)), // -> 128 x 13
                torch::nn::Functional(torch::relu),
                torch::nn::BatchNorm(SCALE * 64),
                torch::nn::MaxPool1d(13) // -> 128 x 1
        ))),
        dense(register_module("dense", torch::nn::Sequential(
                torch::nn::Linear(SCALE * 64 + 6, 6),
                torch::nn::Dropout(0.1)
        ))),
        outputFunction(register_module("outputFunction", torch::nn::Functional(torch::relu))) {}

torch::Tensor RCNNImpl::forward(const torch::Tensor& input, const torch::Tensor& prev) {
    auto t = features->forward(input);
    t = t.transpose(2, 1);
    t = torch::cat({t, prev}, 2);
    t = dense->forward(t);
    return torch::log_softmax(t, 2);
}*/

RCNNImpl::RCNNImpl(int input_size, int channels) :
        features(register_module("features", torch::nn::Sequential(
                // Input: 2 x 2304
                torch::nn::Conv1d(
                        torch::nn::Conv1dOptions(2, SCALE * 16, 3).stride(3)
                ), // -> 16 x 768
                torch::nn::Functional(torch::relu),
                torch::nn::BatchNorm(SCALE * 16),
                // ---------- BLOCK 1 ---------------
                torch::nn::Conv1d(
                        torch::nn::Conv1dOptions(SCALE * 16, SCALE * 16, 3)
                ), // -> 16 x 766
                torch::nn::Functional(torch::relu),
                torch::nn::BatchNorm(SCALE * 16),
                torch::nn::MaxPool1d(3), // -> 16 x 255
                torch::nn::Conv1d(
                        torch::nn::Conv1dOptions(SCALE * 16, SCALE * 32, 3)
                ), // -> 32 x 253
                torch::nn::Functional(torch::relu),
                torch::nn::BatchNorm(SCALE * 32),
                torch::nn::MaxPool1d(3), // -> 32 x 84
                // ---------------------------------
                // ---------- BLOCK 2 ---------------
                torch::nn::Conv1d(
                        torch::nn::Conv1dOptions(SCALE * 32, SCALE * 32, 3)
                ), // -> 32 x 82
                torch::nn::Functional(torch::relu),
                torch::nn::BatchNorm(SCALE * 32),
                torch::nn::MaxPool1d(3), // -> 32 x 27
                // ---------------------------------
                torch::nn::Conv1d(
                        torch::nn::Conv1dOptions(SCALE * 32, SCALE * 64, 3)
                ), // -> 64 x 25
                torch::nn::Functional(torch::relu),
                torch::nn::BatchNorm(SCALE * 64),
                torch::nn::MaxPool1d(25) // -> 64 x 1
        ))),
        dense(register_module("dense", torch::nn::Sequential(
                torch::nn::Linear(SCALE * 64, 6),
                torch::nn::Dropout(0.1)
        ))),
        outputFunction(register_module("outputFunction", torch::nn::Functional(torch::relu))) {}

torch::Tensor RCNNImpl::forward(const torch::Tensor& input) {
    auto t = features->forward(input);
    t = t.transpose(2, 1);
    t = dense->forward(t);
    return torch::log_softmax(t, 2);
}