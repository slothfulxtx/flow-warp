#include <torch/extension.h>
#include "forward_warp.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_warp", &ForwardWarpCUDA);
  m.def("forward_warp_backward", &ForwardWarpBackwardCUDA);
}