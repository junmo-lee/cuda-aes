#include <cuda_runtime.h>
#include <stdint.h>
#include "bs_sbox.cuh"
#include "bs_helpers.cuh"

/**
 * @brief CUDA kernel to perform only the S-box transformation on bitsliced data.
 * 
 * @param in_slices Pointer to the input bitsliced data.
 * @param out_slices Pointer to the output bitsliced data.
 * @param groups Number of 128-slice groups to process.
 */
__global__
void aes_bs_sbox_only(const uint32_t* __restrict__ in_slices,
                      uint32_t* __restrict__ out_slices,
                      unsigned long long groups)
{
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= groups) return;
    const uint32_t* in  = in_slices  + tid * 128ull;
    uint32_t*       out = out_slices + tid * 128ull;
    uint32_t r[128];
    load128<0>(in, r);
    apply_sbox_bytes<0>(r);
    store128<0>(r, out);
}
