// =============================================================================
// aes_bruteforce_bs_kernel.cu  –  Bitsliced AES-128 brute-force kernel
//
// Strategy:  Each CUDA thread tests 32 consecutive candidate keys in parallel
//            using the bitslice technique: one uint32 "lane" per key.
//
// Data layout (per thread):
//   r[128]  – AES state: r[byte*8+bit] has bit j = bit<bit> of byte<byte>
//              for the plaintext/ciphertext of lane j.
//   rk[128] – round key in the same bitsliced format.
//
// Because the same plaintext is used for every lane, all bits of r[] start as
// either 0x00000000 or 0xFFFFFFFF.  The 32 keys differ, so r[] diverges after
// the first AddRoundKey.
// =============================================================================

#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

#include "bs_sbox.cuh"
#include "bs_mixcol.cuh"
#include "bs_helpers.cuh"
#include "bs_keyschedule.cuh"

// ── Local copies of helpers (defined in aes_full_kernel.cu but not exported) ─

__device__ __forceinline__ void shiftrows_bf(uint32_t (&r)[128])
{
    uint32_t t;
    // row 1: rotate left by 1  (1→5→9→13→1)
    for (int k = 0; k < 8; k++) {
        t = r[1*8+k]; r[1*8+k] = r[5*8+k]; r[5*8+k] = r[9*8+k];
        r[9*8+k] = r[13*8+k]; r[13*8+k] = t;
    }
    // row 2: rotate left by 2  (swap pairs: 2↔10, 6↔14)
    for (int k = 0; k < 8; k++) {
        t = r[2*8+k]; r[2*8+k] = r[10*8+k]; r[10*8+k] = t;
        t = r[6*8+k]; r[6*8+k] = r[14*8+k]; r[14*8+k] = t;
    }
    // row 3: rotate left by 3 = right by 1  (15→11→7→3→15)
    for (int k = 0; k < 8; k++) {
        t = r[15*8+k]; r[15*8+k] = r[11*8+k]; r[11*8+k] = r[7*8+k];
        r[7*8+k] = r[3*8+k]; r[3*8+k] = t;
    }
}

template<int BYTE>
__device__ __forceinline__ void subbytes_bf(uint32_t (&r)[128])
{
    constexpr int base = BYTE * 8;
    bs_sbox(r[base+0], r[base+1], r[base+2], r[base+3],
            r[base+4], r[base+5], r[base+6], r[base+7]);
    if constexpr (BYTE + 1 < 16) subbytes_bf<BYTE+1>(r);
}

// XOR the entire state with a round key (both in bitsliced form)
__device__ __forceinline__ void addkey_bs(uint32_t (&r)[128],
                                          const uint32_t (&rk)[128])
{
    for (int i = 0; i < 128; i++) r[i] ^= rk[i];
}

// -----------------------------------------------------------------------------
// check_match_bs
//
// After encryption, determine which of the 32 lanes produced `expected`.
// Returns a lane bitmask: bit j is set iff lane j's ciphertext == expected.
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t check_match_bs(
        const uint32_t (&r)[128],
        const uint8_t   expected[16])
{
    uint32_t match = 0xFFFFFFFFu;
    for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
        const uint8_t eb = expected[byte_idx];
        for (int b = 0; b < 8; b++) {
            if ((eb >> b) & 1u)
                match &=  r[byte_idx*8 + b];   // lane must have bit = 1
            else
                match &= ~r[byte_idx*8 + b];   // lane must have bit = 0
        }
    }
    return match;
}

// =============================================================================
// aes128_bs32_bruteforce
//
// Parameters
// ----------
// plaintext   : 16-byte known plaintext (device pointer, same for all lanes)
// ciphertext  : 16-byte target ciphertext (device pointer)
// base_key    : 4-word big-endian starting key (inclusive)
// total_keys  : number of candidate keys to test across the whole launch
// found_key   : output – 4 words (big-endian) of the found key
// found_flag  : output – atomically set to 1 when a key is found
// =============================================================================
__global__ void aes128_bs32_bruteforce(
        const uint8_t*  __restrict__ plaintext,
        const uint8_t*  __restrict__ ciphertext,
        const uint32_t* __restrict__ base_key,
        uint64_t  total_keys,
        uint32_t* __restrict__ found_key,
        int*      __restrict__ found_flag)
{
    const uint64_t tid        = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t key_offset = tid * 32ull;
    if (key_offset >= total_keys) return;

    // Early-exit optimisation: another thread may have already found the key.
    if (*found_flag) return;

    // ── Compute base key for this thread's 32-key batch ──────────────────
    const uint64_t blo = ((uint64_t)base_key[2] << 32) | base_key[3];
    const uint64_t bhi = ((uint64_t)base_key[0] << 32) | base_key[1];
    const uint64_t lo  = blo + key_offset;
    const uint64_t hi  = bhi + (lo < key_offset ? 1ull : 0ull);

    const uint32_t batch_base[4] = {
        (uint32_t)(hi >> 32),
        (uint32_t)(hi & 0xFFFFFFFFu),
        (uint32_t)(lo >> 32),
        (uint32_t)(lo & 0xFFFFFFFFu)
    };

    // ── Initialise bitsliced state from plaintext ─────────────────────────
    // All 32 lanes encrypt the same plaintext, so each slice is either all-0
    // or all-1.
    uint32_t r[128];
    for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
        const uint8_t pb = plaintext[byte_idx];
        for (int b = 0; b < 8; b++)
            r[byte_idx*8 + b] = ((pb >> b) & 1u) ? 0xFFFFFFFFu : 0x00000000u;
    }

    // ── Initialise bitsliced round-key-0 from 32 consecutive keys ─────────
    uint32_t rk[128];
    bs_ks_init_from_counter(rk, batch_base);

    // ── Round 0: AddRoundKey ──────────────────────────────────────────────
    addkey_bs(r, rk);

    // ── Rounds 1..9: SubBytes → ShiftRows → MixColumns → ExpandKey → ARK ─
    #pragma unroll
    for (int rnd = 1; rnd <= 9; rnd++) {
        subbytes_bf<0>(r);
        shiftrows_bf(r);
        apply_mixcol<0,1,2,3>(r);
        apply_mixcol<4,5,6,7>(r);
        apply_mixcol<8,9,10,11>(r);
        apply_mixcol<12,13,14,15>(r);
        bs_ks_expand_inplace(rk, rnd);
        addkey_bs(r, rk);
    }

    // ── Round 10: SubBytes → ShiftRows → ExpandKey → ARK (no MixColumns) ─
    subbytes_bf<0>(r);
    shiftrows_bf(r);
    bs_ks_expand_inplace(rk, 10);
    addkey_bs(r, rk);

    // ── Match detection ───────────────────────────────────────────────────
    uint32_t match = check_match_bs(r, ciphertext);

    // Mask out lanes that are beyond the requested key range.
    const uint64_t remaining = total_keys - key_offset;
    if (remaining < 32u) {
        const uint32_t valid = (1u << (uint32_t)remaining) - 1u;
        match &= valid;
    }

    if (!match) return;

    // ── Report the first matching lane ───────────────────────────────────
    while (match) {
        const int lane = __ffs(match) - 1;   // index of lowest set bit

        // Reconstruct the full key for this lane.
        const uint64_t klo = ((uint64_t)batch_base[2] << 32) | batch_base[3];
        const uint64_t khi = ((uint64_t)batch_base[0] << 32) | batch_base[1];
        const uint64_t klo2 = klo + (uint64_t)lane;
        const uint64_t khi2 = khi + (klo2 < (uint64_t)lane ? 1ull : 0ull);

        // Only one thread writes the result (first to win the CAS).
        if (atomicCAS(found_flag, 0, 1) == 0) {
            found_key[0] = (uint32_t)(khi2 >> 32);
            found_key[1] = (uint32_t)(khi2 & 0xFFFFFFFFu);
            found_key[2] = (uint32_t)(klo2 >> 32);
            found_key[3] = (uint32_t)(klo2 & 0xFFFFFFFFu);
        }

        match &= match - 1u;   // clear lowest set bit
    }
}


// =============================================================================
// Host-side launcher
// =============================================================================

#define CUDA_OK(call) do {                                               \
    cudaError_t _e = (call);                                             \
    if (_e != cudaSuccess) {                                             \
        std::fprintf(stderr, "%s:%d CUDA error: %s\n",                  \
                     __FILE__, __LINE__, cudaGetErrorString(_e));        \
        return -1;                                                       \
    }                                                                    \
} while(0)

// -----------------------------------------------------------------------------
// run_aes_bs_bruteforce
//
// Launch the bitsliced brute-force kernel for the key range [key_start, key_end).
//
// Returns:
//   1  – key found; found_key[4] is filled with the 4-word big-endian key.
//   0  – key not found in the given range.
//  -1  – CUDA error.
// -----------------------------------------------------------------------------
extern "C" int run_aes_bs_bruteforce(
        const uint8_t  plaintext[16],
        const uint8_t  ciphertext[16],
        const uint32_t key_start[4],
        const uint32_t key_end[4],
        uint32_t       found_key[4],
        dim3 grid, dim3 block)
{
    // Key count (lower-64-bit range; matches existing Key128 iteration logic).
    const uint64_t ks_lo = ((uint64_t)key_start[2] << 32) | key_start[3];
    const uint64_t ke_lo = ((uint64_t)key_end[2]   << 32) | key_end[3];
    const uint64_t total_keys = ke_lo - ks_lo;
    if (total_keys == 0) return 0;

    // Allocate device buffers.
    uint8_t  *d_pt = nullptr, *d_ct = nullptr;
    uint32_t *d_bk = nullptr, *d_fk = nullptr;
    int      *d_ff = nullptr;

    CUDA_OK(cudaMalloc(&d_pt, 16));
    CUDA_OK(cudaMalloc(&d_ct, 16));
    CUDA_OK(cudaMalloc(&d_bk, 16));
    CUDA_OK(cudaMalloc(&d_fk, 16));
    CUDA_OK(cudaMalloc(&d_ff, sizeof(int)));

    CUDA_OK(cudaMemcpy(d_pt, plaintext,  16, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_ct, ciphertext, 16, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_bk, key_start,  16, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(d_ff, 0, sizeof(int)));

    // Time the kernel.
    cudaEvent_t ev0, ev1;
    CUDA_OK(cudaEventCreate(&ev0));
    CUDA_OK(cudaEventCreate(&ev1));
    CUDA_OK(cudaEventRecord(ev0));

    aes128_bs32_bruteforce<<<grid, block>>>(
        d_pt, d_ct, d_bk, total_keys, d_fk, d_ff);

    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaEventRecord(ev1));
    CUDA_OK(cudaEventSynchronize(ev1));

    float ms = 0.f;
    CUDA_OK(cudaEventElapsedTime(&ms, ev0, ev1));
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    // Retrieve results.
    int      h_ff = 0;
    uint32_t h_fk[4] = {0, 0, 0, 0};
    CUDA_OK(cudaMemcpy(&h_ff, d_ff, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_ff)
        CUDA_OK(cudaMemcpy(h_fk, d_fk, 16, cudaMemcpyDeviceToHost));

    cudaFree(d_pt); cudaFree(d_ct); cudaFree(d_bk);
    cudaFree(d_fk); cudaFree(d_ff);

    // Throughput report.
    const uint64_t threads_launched = (uint64_t)grid.x * block.x;
    const uint64_t threads_active   = (threads_launched * 32u <= total_keys)
                                      ? threads_launched
                                      : (total_keys + 31u) / 32u;
    const uint64_t keys_tested      = threads_active * 32u;
    const double   secs             = (double)ms * 1e-3;
    const double   mkps             = secs > 0.0 ? keys_tested / secs / 1e6 : 0.0;

    std::printf("[bs-bf] %.3f ms | keys=%llu (32/thread) | %.1f Mkeys/s\n",
                ms, (unsigned long long)keys_tested, mkps);

    if (h_ff) {
        found_key[0] = h_fk[0]; found_key[1] = h_fk[1];
        found_key[2] = h_fk[2]; found_key[3] = h_fk[3];
        return 1;
    }
    return 0;
}
