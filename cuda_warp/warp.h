#ifndef CUDA_WARP_H_INCLUDED
#define CUDA_WARP_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaWarp
{
  class Warp
  {
  public:

    static void forward(
      const int B,
      const int C,
      const int H,
      const int W,
      const float* x,
      const float* flow,
      const int mode,
      float* warped_x);

    static void backward(
      const int B,
      const int C,
      const int H,
      const int W,
      const float* x,
      const float* flow,
      const int mode,
      const float* dL_dout,
      float* dL_dx,
      float* dL_dflow);
  };
};

#endif