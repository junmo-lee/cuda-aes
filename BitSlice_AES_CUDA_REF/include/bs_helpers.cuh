#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include "bs_sbox.cuh"

template<int I>
__device__ __forceinline__ void load128(const uint32_t* __restrict__ base,
                                        uint32_t (&r)[128]) {
    r[I] = base[I];
    if constexpr (I + 1 < 128) load128<I+1>(base, r);
}
template<int I>
__device__ __forceinline__ void store128(const uint32_t (&r)[128],
                                         uint32_t* __restrict__ base) {
    base[I] = r[I];
    if constexpr (I + 1 < 128) store128<I+1>(r, base);
}
template<int BYTE>
__device__ __forceinline__ void apply_sbox_bytes(uint32_t (&r)[128]) {
    constexpr int base = BYTE * 8;
    bs_sbox(r[base+0], r[base+1], r[base+2], r[base+3],
            r[base+4], r[base+5], r[base+6], r[base+7]);
    if constexpr (BYTE + 1 < 16) apply_sbox_bytes<BYTE+1>(r);
}
