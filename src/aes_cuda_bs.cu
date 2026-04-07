// ============================================================================
// aes_cuda_bs.cu – Bitsliced AES-128 CUDA brute-force engine
//
// Kernel adapted from BitSlice_AES_CUDA_REF (same repository).
// Strategy: each CUDA thread tests 32 consecutive candidate keys in parallel
// using the bitslice technique — one uint32 "lane" per key.
//
// Data layout (per thread):
//   r[128]  – AES state: r[byte*8+bit] holds bit j = bit<bit> of byte<byte>
//             for lane j.
//   rk[128] – round key in the same bitsliced format.
//
// All 32 lanes encrypt the same plaintext with distinct keys, so each slice
// of r[] starts as either 0x00000000 or 0xFFFFFFFF.
// ============================================================================

#include "aes_cuda_bs.cuh"
#include "logger.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstring>
#include <chrono>
#include <stdexcept>

// Bitslice library headers (from BitSlice_AES_CUDA_REF/include/)
#include "bs_sbox.cuh"
#include "bs_mixcol.cuh"
#include "bs_helpers.cuh"
#include "bs_keyschedule.cuh"

// ── Error helper ──────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            char _buf[256];                                                 \
            snprintf(_buf, sizeof(_buf), "CUDA error %s:%d – %s",          \
                     __FILE__, __LINE__, cudaGetErrorString(_e));           \
            throw std::runtime_error(_buf);                                 \
        }                                                                   \
    } while (0)

// ── Kernel parameters ─────────────────────────────────────────────────────────
#define BS_BLOCKS   1024
#define BS_THREADS   256
#define BS_KEYS_PER_THREAD 32ULL

// ── Bitsliced helpers (local to this TU) ──────────────────────────────────────

/**
 * @brief Performs the AES ShiftRows operation in bitsliced format.
 * 
 * Operates on a 32-lane bitsliced state, shifting the bytes according to the
 * AES specification.
 * 
 * @param r The 128-word bitsliced AES state (32 lanes).
 */
__device__ __forceinline__ void bs_shiftrows(uint32_t (&r)[128])
{
    uint32_t t;
    // row 1: rotate left by 1  (bytes 1→5→9→13→1)
    for (int k = 0; k < 8; k++) {
        t = r[1*8+k]; r[1*8+k] = r[5*8+k]; r[5*8+k] = r[9*8+k];
        r[9*8+k] = r[13*8+k]; r[13*8+k] = t;
    }
    // row 2: rotate left by 2  (swap pairs: 2↔10, 6↔14)
    for (int k = 0; k < 8; k++) {
        t = r[2*8+k];  r[2*8+k]  = r[10*8+k]; r[10*8+k] = t;
        t = r[6*8+k];  r[6*8+k]  = r[14*8+k]; r[14*8+k] = t;
    }
    // row 3: rotate left by 3 = right by 1  (15→11→7→3→15)
    for (int k = 0; k < 8; k++) {
        t = r[15*8+k]; r[15*8+k] = r[11*8+k]; r[11*8+k] = r[7*8+k];
        r[7*8+k] = r[3*8+k]; r[3*8+k] = t;
    }
}

/**
 * @brief Performs the AES SubBytes operation in bitsliced format for a single byte.
 * 
 * Uses a compile-time template to iterate through all 16 bytes of the state.
 * 
 * @tparam BYTE The current byte index (0-15) being processed.
 * @param r The 128-word bitsliced AES state (32 lanes).
 */
template<int BYTE>
__device__ __forceinline__ void bs_subbytes(uint32_t (&r)[128])
{
    constexpr int base = BYTE * 8;
    bs_sbox(r[base+0], r[base+1], r[base+2], r[base+3],
            r[base+4], r[base+5], r[base+6], r[base+7]);
    if constexpr (BYTE + 1 < 16) bs_subbytes<BYTE+1>(r);
}

/**
 * @brief Performs the AES AddRoundKey operation in bitsliced format.
 * 
 * XORs the bitsliced round key into the bitsliced state.
 * 
 * @param r The 128-word bitsliced AES state (32 lanes).
 * @param rk The 128-word bitsliced round key.
 */
__device__ __forceinline__ void bs_addkey(uint32_t (&r)[128],
                                          const uint32_t (&rk)[128])
{
    for (int i = 0; i < 128; i++) r[i] ^= rk[i];
}

/**
 * @brief Checks for a match between the bitsliced state and the expected ciphertext.
 * 
 * Compares each of the 32 lanes in the bitsliced state against the target
 * ciphertext block.
 * 
 * @param r The 128-word bitsliced AES state (32 lanes).
 * @param expected The 16-byte target ciphertext block.
 * @return A 32-bit mask where bit j is set if lane j matches the expected ciphertext.
 */
__device__ __forceinline__ uint32_t bs_check_match(
        const uint32_t (&r)[128],
        const uint8_t   expected[16])
{
    uint32_t match = 0xFFFFFFFFu;
    for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
        const uint8_t eb = expected[byte_idx];
        for (int b = 0; b < 8; b++) {
            if ((eb >> b) & 1u)
                match &=  r[byte_idx*8 + b];
            else
                match &= ~r[byte_idx*8 + b];
        }
    }
    return match;
}

/**
 * @brief Prints the AES state for debugging (lane 0 only).
 * 
 * @param label A label to identify the state being printed.
 * @param round The current AES round number.
 * @param r The 128-word bitsliced AES state.
 */
__device__ void debug_print_state(const char* label, int round, const uint32_t r[128]) {
    uint8_t out[16];
    for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
        uint8_t pb = 0;
        for (int b = 0; b < 8; b++) {
            if (r[byte_idx * 8 + b] & 1u) {
                pb |= (1u << b);
            }
        }
        out[byte_idx] = pb;
    }
    printf("round[%2d].%-5s ", round, label);
    for (int i = 0; i < 16; i++) printf("%02x", out[i]);
    printf("\n");
}

/**
 * @brief Prints the round key for debugging (lane 0 only).
 * 
 * @param label A label to identify the key being printed.
 * @param round The current AES round number.
 * @param rk The 128-word bitsliced round key.
 */
__device__ void debug_print_rk(const char* label, int round, const uint32_t rk[128]) {
    uint8_t out[16];
    for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
        uint8_t pb = 0;
        for (int b = 0; b < 8; b++) {
            if (rk[byte_idx * 8 + b] & 1u) {
                pb |= (1u << b);
            }
        }
        out[byte_idx] = pb;
    }
    printf("round[%2d].%-5s ", round, label);
    for (int i = 0; i < 16; i++) printf("%02x", out[i]);
    printf("\n");
}

// ── Bitsliced AES-128 brute-force kernel ──────────────────────────────────────
/**
 * @brief Bitsliced AES-128 brute-force kernel.
 * 
 * Each thread tests 32 consecutive keys in parallel using bitslicing.
 * 
 * @param plaintext Known 16-byte plaintext (device pointer).
 * @param ciphertext Target 16-byte ciphertext (device pointer).
 * @param base_key Starting 128-bit key for this kernel launch (4x32-bit).
 * @param total_keys Total number of keys to test in this launch.
 * @param found_key Device pointer to store the 128-bit key if found.
 * @param found_flag Device pointer to an integer flag set when key is found.
 */
__global__ void aes128_bs32_bruteforce_kernel(
        const uint8_t*  __restrict__ plaintext,
        const uint8_t*  __restrict__ ciphertext,
        const uint32_t* __restrict__ base_key,
        uint64_t  total_keys,
        uint32_t* __restrict__ found_key,
        int*      __restrict__ found_flag)
{
    const uint64_t tid        = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t key_offset = tid * BS_KEYS_PER_THREAD;
    if (key_offset >= total_keys) return;

    // Early exit if another thread already found the key.
    if (*found_flag) return;

    bool do_print = (tid == 0);

    // ── Compute base key for this thread's 32-key batch ───────────────────
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

    // ── Initialise bitsliced state from plaintext ──────────────────────────
    // All 32 lanes share the same plaintext; each bit-slice is all-0 or all-1.
    uint32_t r[128];
    for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
        const uint8_t pb = plaintext[byte_idx];
        for (int b = 0; b < 8; b++)
            r[byte_idx*8 + b] = ((pb >> b) & 1u) ? 0xFFFFFFFFu : 0x00000000u;
    }

    if (do_print) debug_print_state("input", 0, r);

    // ── Initialise bitsliced round-key-0 from 32 consecutive keys ─────────
    uint32_t rk[128];
    bs_ks_init_from_counter(rk, batch_base);

    if (do_print) debug_print_rk("k_sch", 0, rk);

    // ── Round 0: AddRoundKey ───────────────────────────────────────────────
    bs_addkey(r, rk);

    // ── Rounds 1..9: SubBytes → ShiftRows → MixColumns → ExpandKey → ARK ──
    #pragma unroll
    for (int rnd = 1; rnd <= 9; rnd++) {
        if (do_print) debug_print_state("start", rnd, r);
        bs_subbytes<0>(r);
        if (do_print) debug_print_state("s_box", rnd, r);
        bs_shiftrows(r);
        if (do_print) debug_print_state("s_row", rnd, r);
        apply_mixcol<0,1,2,3>(r, do_print);
        apply_mixcol<4,5,6,7>(r, do_print);
        apply_mixcol<8,9,10,11>(r, do_print);
        apply_mixcol<12,13,14,15>(r, do_print);
        if (do_print) debug_print_state("m_col", rnd, r);
        bs_ks_expand_inplace(rk, rnd);
        if (do_print) debug_print_rk("k_sch", rnd, rk);
        bs_addkey(r, rk);
    }

    // ── Round 10: SubBytes → ShiftRows → ExpandKey → ARK (no MixColumns) ──
    if (do_print) debug_print_state("start", 10, r);
    bs_subbytes<0>(r);
    if (do_print) debug_print_state("s_box", 10, r);
    bs_shiftrows(r);
    if (do_print) debug_print_state("s_row", 10, r);
    bs_ks_expand_inplace(rk, 10);
    if (do_print) debug_print_rk("k_sch", 10, rk);
    bs_addkey(r, rk);

    if (do_print) debug_print_state("output", 10, r);

    // ── Match detection ───────────────────────────────────────────────────
    uint32_t match = bs_check_match(r, ciphertext);

    // Mask out lanes beyond the requested key range.
    const uint64_t remaining = total_keys - key_offset;
    if (remaining < BS_KEYS_PER_THREAD) {
        const uint32_t valid = (1u << (uint32_t)remaining) - 1u;
        match &= valid;
    }

    if (!match) return;

    // ── Report the first matching lane ────────────────────────────────────
    while (match) {
        const int lane = __ffs(match) - 1;

        // Reconstruct full 128-bit key for this lane.
        const uint64_t klo = ((uint64_t)batch_base[2] << 32) | batch_base[3];
        const uint64_t khi = ((uint64_t)batch_base[0] << 32) | batch_base[1];
        const uint64_t klo2 = klo + (uint64_t)lane;
        const uint64_t khi2 = khi + (klo2 < (uint64_t)lane ? 1ull : 0ull);

        if (atomicCAS(found_flag, 0, 1) == 0) {
            found_key[0] = (uint32_t)(khi2 >> 32);
            found_key[1] = (uint32_t)(khi2 & 0xFFFFFFFFu);
            found_key[2] = (uint32_t)(klo2 >> 32);
            found_key[3] = (uint32_t)(klo2 & 0xFFFFFFFFu);
        }

        match &= match - 1u;  // clear lowest set bit
    }
}

// ── Host-side GPU brute-force (bitsliced) ────────────────────────────────────
/**
 * @brief Host-side wrapper for bitsliced GPU brute-force search.
 * 
 * Manages device memory, kernel launches, and result retrieval for the 
 * bitsliced AES brute-force engine.
 * 
 * @param device_id The CUDA device index to use.
 * @param plaintext Known plaintext block.
 * @param ciphertext Target ciphertext block.
 * @param key_start Starting key for the search.
 * @param num_keys Number of keys to test.
 * @param found_key Output parameter for the found key.
 * @param stop_flag Atomic flag to signal early termination.
 * @param keys_tried Atomic counter for tracking progress.
 * @return True if key was found, false otherwise.
 */
bool gpu_bruteforce_bs(
    int                device_id,
    const Block128&    plaintext,
    const Block128&    ciphertext,
    Key128             key_start,
    u64                num_keys,
    Key128&            found_key,
    std::atomic<bool>& stop_flag,
    std::atomic<u64>&  keys_tried
) {
    CUDA_CHECK(cudaSetDevice(device_id));

    // Allocate device buffers (uint8_t for pt/ct, uint32_t for keys).
    uint8_t  *d_pt, *d_ct;
    uint32_t *d_bk, *d_fk;
    int      *d_ff;
    CUDA_CHECK(cudaMalloc(&d_pt, 16));
    CUDA_CHECK(cudaMalloc(&d_ct, 16));
    CUDA_CHECK(cudaMalloc(&d_bk, 16));
    CUDA_CHECK(cudaMalloc(&d_fk, 16));
    CUDA_CHECK(cudaMalloc(&d_ff, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ff, 0, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_pt, plaintext.data,  16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ct, ciphertext.data, 16, cudaMemcpyHostToDevice));

    const u64 keys_per_launch = (u64)BS_BLOCKS * BS_THREADS * BS_KEYS_PER_THREAD;

    bool result   = false;
    u64  remaining = num_keys;
    u64  lo64     = ((u64)key_start.w[2] << 32) | key_start.w[3];
    u32  rk0      = key_start.w[0];
    u32  rk1      = key_start.w[1];

    while (remaining > 0 && !stop_flag.load(std::memory_order_relaxed)) {
        u64 batch = (remaining < keys_per_launch) ? remaining : keys_per_launch;

        // Upload base key for this batch.
        uint32_t h_bk[4] = {
            rk0, rk1,
            (u32)(lo64 >> 32),
            (u32)(lo64 & 0xFFFFFFFFULL)
        };
        CUDA_CHECK(cudaMemcpy(d_bk, h_bk, 16, cudaMemcpyHostToDevice));

        // Grid: one thread per 32 keys.
        u64 threads_needed = (batch + BS_KEYS_PER_THREAD - 1) / BS_KEYS_PER_THREAD;
        int blocks_needed  = (int)((threads_needed + BS_THREADS - 1) / BS_THREADS);
        if (blocks_needed < 1) blocks_needed = 1;

        aes128_bs32_bruteforce_kernel<<<blocks_needed, BS_THREADS>>>(
            d_pt, d_ct, d_bk, batch, d_fk, d_ff);
        CUDA_CHECK(cudaDeviceSynchronize());

        keys_tried.fetch_add(batch, std::memory_order_relaxed);
        remaining -= batch;

        // Advance base key (128-bit addition).
        u64 new_lo = lo64 + batch;
        if (new_lo < lo64) {           // carry into upper 64 bits
            u64 hi64 = ((u64)rk0 << 32) | rk1;
            hi64++;
            rk0 = (u32)(hi64 >> 32);
            rk1 = (u32)(hi64 & 0xFFFFFFFFULL);
        }
        lo64 = new_lo;

        int h_ff = 0;
        CUDA_CHECK(cudaMemcpy(&h_ff, d_ff, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_ff) {
            uint32_t h_fk[4];
            CUDA_CHECK(cudaMemcpy(h_fk, d_fk, 16, cudaMemcpyDeviceToHost));
            found_key = Key128(h_fk[0], h_fk[1], h_fk[2], h_fk[3]);
            stop_flag.store(true, std::memory_order_relaxed);
            result = true;
            break;
        }
    }

    cudaFree(d_pt); cudaFree(d_ct); cudaFree(d_bk);
    cudaFree(d_fk); cudaFree(d_ff);
    return result;
}

// ── Benchmark (bitsliced) ─────────────────────────────────────────────────────
/**
 * @brief Benchmarks the bitsliced GPU AES performance.
 * 
 * @param device_id The CUDA device index to benchmark.
 * @param duration_ms Target duration for the benchmark in milliseconds.
 * @return Measured performance in keys per second.
 */
double gpu_benchmark_bs(int device_id, int duration_ms) {
    CUDA_CHECK(cudaSetDevice(device_id));

    uint8_t  *d_pt, *d_ct;
    uint32_t *d_bk, *d_fk;
    int      *d_ff;
    CUDA_CHECK(cudaMalloc(&d_pt, 16));
    CUDA_CHECK(cudaMalloc(&d_ct, 16));
    CUDA_CHECK(cudaMalloc(&d_bk, 16));
    CUDA_CHECK(cudaMalloc(&d_fk, 16));
    CUDA_CHECK(cudaMalloc(&d_ff, sizeof(int)));

    uint8_t  h_pt[16] = {0}, h_ct[16] = {0};
    uint32_t h_bk[4]  = {0, 0, 0, 0};
    CUDA_CHECK(cudaMemcpy(d_pt, h_pt, 16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ct, h_ct, 16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bk, h_bk, 16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_ff, 0, sizeof(int)));

    const u64 keys_per_launch = (u64)BS_BLOCKS * BS_THREADS * BS_KEYS_PER_THREAD;

    auto t0    = std::chrono::high_resolution_clock::now();
    u64  count = 0;

    while (true) {
        aes128_bs32_bruteforce_kernel<<<BS_BLOCKS, BS_THREADS>>>(
            d_pt, d_ct, d_bk, keys_per_launch, d_fk, d_ff);
        cudaDeviceSynchronize();
        count += keys_per_launch;

        auto   now = std::chrono::high_resolution_clock::now();
        double ms  = std::chrono::duration<double, std::milli>(now - t0).count();
        if (ms >= duration_ms) {
            cudaFree(d_pt); cudaFree(d_ct); cudaFree(d_bk);
            cudaFree(d_fk); cudaFree(d_ff);
            return count / (ms / 1000.0);
        }
    }
}
