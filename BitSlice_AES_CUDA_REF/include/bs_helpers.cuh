#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include "bs_sbox.cuh"

/**
 * @brief Recursively loads 128 bits (as 128 32-bit words in bitsliced format) from memory.
 * 
 * @tparam I The current bit index being loaded (starts from 0).
 * @param base Pointer to the source memory.
 * @param r Reference to the array where bits are loaded.
 * @return void
 */
template<int I>
__device__ __forceinline__ void load128(const uint32_t* __restrict__ base,
                                        uint32_t (&r)[128]) {
    r[I] = base[I];
    if constexpr (I + 1 < 128) load128<I+1>(base, r);
}

/**
 * @brief Recursively stores 128 bits (as 128 32-bit words in bitsliced format) to memory.
 * 
 * @tparam I The current bit index being stored (starts from 0).
 * @param r Reference to the array containing the bits.
 * @param base Pointer to the destination memory.
 * @return void
 */
template<int I>
__device__ __forceinline__ void store128(const uint32_t (&r)[128],
                                         uint32_t* __restrict__ base) {
    base[I] = r[I];
    if constexpr (I + 1 < 128) store128<I+1>(r, base);
}

/**
 * @brief Recursively applies the bitsliced S-box to all 16 bytes of the AES state.
 * 
 * @tparam BYTE The current byte index being processed (0-15).
 * @param r Reference to the 128-bit state in bitsliced format.
 * @return void
 */
template<int BYTE>
__device__ __forceinline__ void apply_sbox_bytes(uint32_t (&r)[128]) {
    constexpr int base = BYTE * 8;
    bs_sbox(r[base+0], r[base+1], r[base+2], r[base+3],
            r[base+4], r[base+5], r[base+6], r[base+7]);
    if constexpr (BYTE + 1 < 16) apply_sbox_bytes<BYTE+1>(r);
}
