// src/aes_debug_kernels.cu
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__
void aes_io_identity(const uint32_t* __restrict__ in_slices,
                     uint32_t* __restrict__ out_slices,
                     unsigned long long groups) {
  unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= groups) return;
  const uint32_t* in  = in_slices  + tid*128ull;
  uint32_t*       out = out_slices + tid*128ull;
  #pragma unroll
  for (int i = 0; i < 128; ++i) out[i] = in[i];
}
