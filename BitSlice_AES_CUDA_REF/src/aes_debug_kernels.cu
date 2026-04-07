// src/aes_debug_kernels.cu
#include <cuda_runtime.h>
#include <stdint.h>

/**
 * @brief Simple CUDA kernel to copy bitsliced data from input to output.
 * 
 * This serves as a debug or identity kernel to verify the bitslicing pipeline.
 * 
 * @param in_slices Pointer to the input bitsliced data.
 * @param out_slices Pointer to the output bitsliced data.
 * @param groups Number of 128-slice groups to process.
 */
extern "C" __global__
void aes_io_identity(const uint32_t* __restrict__ in_slices,
                     uint32_t* __restrict__ out_slices,
                     unsigned long long groups) {
  unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= groups) return;
  const uint32_t* in  = in_slices  + tid*128ull;
  uint32_t*       out = out_slices + tid*128ull;
  #pragma unroll
  for (int i = 0; i < 128; ++i) out[i] = in[i];
}
