#include <cuda_runtime.h>
#include <stdint.h>
#include "bs_helpers.cuh"
#include "bs_mixcol.cuh"

__global__
void aes_bs_mix_only(const uint32_t* __restrict__ in_slices,
                     uint32_t* __restrict__ out_slices,
                     unsigned long long groups)
{
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= groups) return;
    const uint32_t* in  = in_slices  + tid * 128ull;
    uint32_t*       out = out_slices + tid * 128ull;
    uint32_t r[128];
    load128<0>(in, r);
    apply_mixcol<0,4,8,12>(r);
    apply_mixcol<1,5,9,13>(r);
    apply_mixcol<2,6,10,14>(r);
    apply_mixcol<3,7,11,15>(r);
    store128<0>(r, out);
}

