#pragma once


#include <torch/torch.h>

#define SCALE 1

/**
 * A CNN implementation for spoken language classification
 */
struct CNNImpl : torch::nn::Module {
    CNNImpl();

    torch::Tensor forward(const torch::Tensor& input);

    torch::nn::Sequential features;
    torch::nn::Sequential dense;
    torch::nn::Functional outputFunction;
};

TORCH_MODULE(CNN);


