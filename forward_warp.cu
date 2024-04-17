#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_warp/warp.h"
#include <fstream>
#include <string>
#include <functional>


torch::Tensor ForwardWarpCUDA(
  const torch::Tensor& x,
  const torch::Tensor& flow,
  const int mode)
{
  const int B = x.size(0);
  const int C = x.size(1);
  const int H = x.size(2);
  const int W = x.size(3);

  auto int_opts = x.options().dtype(torch::kInt32);
  auto float_opts = x.options().dtype(torch::kFloat32);

  torch::Tensor warped_x = torch::full({B, C, H, W}, 0.0, float_opts);
  
  CudaWarp::Warp::forward(
    B, C, H, W,
    x.contiguous().data<float>(),
    flow.contiguous().data<float>(),
    mode,
    warped_x.contiguous().data<float>()
  );
  return warped_x;
}

std::tuple<torch::Tensor, torch::Tensor> ForwardWarpBackwardCUDA(
  const torch::Tensor& dL_dout,
  const torch::Tensor& x,
  const torch::Tensor& flow,
  const int mode)
{
  const int B = x.size(0);
  const int C = x.size(1);
  const int H = x.size(2);
  const int W = x.size(3);

  auto int_opts = x.options().dtype(torch::kInt32);
  auto float_opts = x.options().dtype(torch::kFloat32);

  torch::Tensor dL_dx = torch::full({B, C, H, W}, 0.0, float_opts);
  torch::Tensor dL_dflow = torch::full({B, H, W, 2}, 0.0, float_opts);
  CudaWarp::Warp::backward(
    B, C, H, W,
    x.contiguous().data<float>(),
    flow.contiguous().data<float>(),
    mode,
    dL_dout.contiguous().data<float>(),
    dL_dx.contiguous().data<float>(),
    dL_dflow.contiguous().data<float>()
  );
  return std::make_tuple(dL_dx, dL_dflow);
}