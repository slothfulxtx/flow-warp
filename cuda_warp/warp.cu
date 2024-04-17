#include "warp.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "config.h"


__forceinline__ __device__ int get_index(
  const int b,
  const int c,
  const int h,
  const int w,
  const int C,
  const int H,
  const int W) 
{
  return b*C*H*W + c*H*W + h*W + w;
}

__global__ void forward_kernel(
  const int B,
  const int C,
  const int H,
  const int W,
  const float* feat,
  const float* flow,
  const int mode,
  float* warped_feat
)
{
  auto idx = cg::this_grid().thread_rank();
  if(idx >= B*H*W) return;
  const int b = idx / (H * W);
  const int h = (idx-b*H*W) / W;
  const int w = idx % W;
  const float x = w + flow[idx*2+0];
  const float y = h + flow[idx*2+1];
  if (mode == 0) {
    const int x_f = static_cast<int>(x);
    const int y_f = static_cast<int>(y);
    const int x_c = x_f + 1;
    const int y_c = y_f + 1;
    if(x_f>=0 && x_c<W && y_f>=0 && y_c<H){
      const float nw_k = (x_c - x) * (y_c - y);
      const float ne_k = (x - x_f) * (y_c - y);
      const float sw_k = (x_c - x) * (y - y_f);
      const float se_k = (x - x_f) * (y - y_f);
      const float* f = feat + get_index(b, 0, h, w, C, H, W);
      float* wf = warped_feat+get_index(b, 0, y_f, x_f, C, H, W);
      for (int c = 0; c < C; ++c, f+=H*W, wf+=H*W){
          atomicAdd(wf,     nw_k*(*f));
          atomicAdd(wf+1,   ne_k*(*f));
          atomicAdd(wf+W,   sw_k*(*f));
          atomicAdd(wf+W+1, se_k*(*f));
      }
    }
  } else if (mode == 1) {
    const int x_f = static_cast<int>(round(x));
    const int y_f = static_cast<int>(round(y));
    if(x_f>=0 && x_f<W && y_f>=0 && y_f<H){
      const float* f = feat+get_index(b, 0, h, w, C, H, W);
      float* wf = warped_feat+get_index(b, 0, y_f, x_f, C, H, W);
      for (int c = 0; c < C; ++c, f += H*W, wf += H*W) {
          atomicAdd(wf, (*f));
      }
    }
  }
}



__global__ void backward_kernel(
  const int B,
  const int C,
  const int H,
  const int W,
  const float* feat,
  const float* flow,
  const int mode,
  const float* dL_dwfeat,
  float* dL_dfeat,
  float* dL_dflow
)
{
  auto idx = cg::this_grid().thread_rank();
  if(idx >= B*H*W) return;
  const int b = idx / (H * W);
  const int h = (idx-b*H*W) / W;
  const int w = idx % W;
  const float x = w + flow[idx*2+0];
  const float y = h + flow[idx*2+1];
  if (mode == 0) {
    const int x_f = static_cast<int>(x);
    const int y_f = static_cast<int>(y);
    const int x_c = x_f + 1;
    const int y_c = y_f + 1;
    if(x_f>=0 && x_c<W && y_f>=0 && y_c<H){
      const float nw_k = (x_c - x) * (y_c - y);
      const float ne_k = (x - x_f) * (y_c - y);
      const float sw_k = (x_c - x) * (y - y_f);
      const float se_k = (x - x_f) * (y - y_f);
      float dL_dflow_x = 0;
      float dL_dflow_y = 0;
      const float* dL_dwf = dL_dwfeat + get_index(b, 0, h, w, C, H, W);
      float* dL_df = dL_dfeat + get_index(b, 0, y_f, x_f, C, H, W);
      for (int c = 0; c < C; ++c, dL_df+=H*W, dL_dwf+=H*W){
        const float nw_grad = (*dL_dwf);
        const float ne_grad = (*dL_dwf+1);
        const float sw_grad = (*dL_dwf+W);
        const float se_grad = (*dL_dwf+W+1);
        atomicAdd(dL_df, nw_k*nw_grad+ne_k*ne_grad+sw_k*sw_grad+se_k*se_grad);
        const float p = feat[get_index(b, c, h, w, C, H, W)];        
        dL_dflow_x -= (y_c-y)*p*nw_grad;
        dL_dflow_y -= (x_c-x)*p*nw_grad;
        dL_dflow_x += (y_c-y)*p*ne_grad;
        dL_dflow_y -= (x-x_f)*p*ne_grad;
        dL_dflow_x -= (y-y_f)*p*sw_grad;
        dL_dflow_y += (x_c-x)*p*sw_grad;
        dL_dflow_x += (y-y_f)*p*se_grad;
        dL_dflow_y += (x-x_f)*p*se_grad;
      }
      atomicAdd(dL_dflow + idx*2, dL_dflow_x);
      atomicAdd(dL_dflow + idx*2+1, dL_dflow_y);
    }
  } else if (mode == 1) {
      const int x_f = static_cast<int>(round(x));
      const int y_f = static_cast<int>(round(y));
      if(x_f>=0 && x_f<W && y_f>=0 && y_f<H){
        float* dL_df = dL_dfeat + get_index(b, 0, h, w, C, H, W);
        const float* dL_dwf = dL_dwfeat + get_index(b, 0, y_f, x_f, C, H, W);
        for (int c = 0; c < C; ++c, dL_df += H*W, dL_dwf += H*W) {
            atomicAdd(dL_df, (*dL_dwf));
        
        }
      }
  }
}

void CudaWarp::Warp::forward(
  const int B,
  const int C,
  const int H,
  const int W,
  const float* x,
  const float* flow,
  const int mode,
  float* warped_x)
{

  forward_kernel << < (B*H*W + BLOCK_SIZE-1) / BLOCK_SIZE, BLOCK_SIZE >> > (
    B,C,H,W,
    x, 
    flow, 
    mode, 
    warped_x
  );
}

void CudaWarp::Warp::backward(
  const int B,
  const int C,
  const int H,
  const int W,
  const float* x,
  const float* flow,
  const int mode,
  const float* dL_dout,
  float* dL_dx,
  float* dL_dflow)
{
  backward_kernel << < (B*H*W + BLOCK_SIZE-1) / BLOCK_SIZE, BLOCK_SIZE >> > (
    B,C,H,W,
    x, 
    flow, 
    mode,
    dL_dout,
    dL_dx,
    dL_dflow
  );
}



