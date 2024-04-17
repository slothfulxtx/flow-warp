#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
  
torch::Tensor ForwardWarpCUDA(
  const torch::Tensor& x,
  const torch::Tensor& flow,
  const int mode);

std::tuple<torch::Tensor, torch::Tensor> ForwardWarpBackwardCUDA(
  const torch::Tensor& dL_dout,
  const torch::Tensor& x,
  const torch::Tensor& flow,
  const int mode);