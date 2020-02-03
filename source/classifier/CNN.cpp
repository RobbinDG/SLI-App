#include "CNN.hpp"

CNNImpl::CNNImpl() :
        features(register_module("features", torch::nn::Sequential(
                // Input: 1 x 22118
                torch::nn::Conv1d(
                        torch::nn::Conv1dOptions(1, SCALE * 16, 3).stride(3)
                ), // -> 16 x 7372
                torch::nn::Functional(torch::relu),
                torch::nn::BatchNorm(SCALE * 16),
                // ---------- BLOCK 1 ---------------
                torch::nn::Conv1d(
                        torch::nn::Conv1dOptions(SCALE * 16, SCALE * 16, 3)
                ), // -> 16 x 7370
                torch::nn::Functional(torch::relu),
                torch::nn::BatchNorm(SCALE * 16),
                torch::nn::MaxPool1d(3), // -> 16 x 2456
                torch::nn::Conv1d(
                        torch::nn::Conv1dOptions(SCALE * 16, SCALE * 16, 3)
                ), // -> 32 x 2454
                torch::nn::Functional(torch::relu),
                torch::nn::BatchNorm(SCALE * 16),
                torch::nn::MaxPool1d(3), // -> 32 x 818
                // ---------------------------------
                torch::nn::Conv1d(
                        torch::nn::Conv1dOptions(SCALE * 16, SCALE * 32, 3)
                ), // -> 32 x 816
                torch::nn::Functional(torch::relu),
                torch::nn::BatchNorm(SCALE * 32),
                torch::nn::MaxPool1d(3), // -> 32 x 272
                // ---------- BLOCK 2 ---------------
                torch::nn::Conv1d(
                        torch::nn::Conv1dOptions(SCALE * 32, SCALE * 32, 3)
                ), // -> 32 x 270
                torch::nn::Functional(torch::relu),
                torch::nn::BatchNorm(SCALE * 32),
                torch::nn::MaxPool1d(3), // -> 32 x 90
                // ---------------------------------
                torch::nn::Conv1d(
                        torch::nn::Conv1dOptions(SCALE * 32, SCALE * 64, 3)
                ), // -> 64 x 88
                torch::nn::Functional(torch::relu),
                torch::nn::BatchNorm(SCALE * 64),
                torch::nn::MaxPool1d(88) // -> 64 x 1
        ))),
        dense(register_module("dense", torch::nn::Sequential(
                torch::nn::Linear(SCALE * 64, 6),
                torch::nn::Dropout(0.1)
        ))),
        outputFunction(register_module("outputFunction", torch::nn::Functional(torch::relu))) {}

torch::Tensor CNNImpl::forward(const torch::Tensor& input) {
    auto t = features->forward(input);
    t = t.transpose(2, 1);
    t = dense->forward(t);
    return torch::log_softmax(t, 2);
}