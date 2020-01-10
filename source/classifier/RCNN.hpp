#pragma once


#include <torch/torch.h>

#define SCALE 1

struct RCNNImpl : torch::nn::Module {
    RCNNImpl();

    torch::Tensor forward(const torch::Tensor& input);

    torch::nn::Sequential features;
    torch::nn::Sequential dense;
    torch::nn::Functional outputFunction;
};

TORCH_MODULE(RCNN);


