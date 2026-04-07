// ============================================================================
// aes_cuda.cu – CUDA AES-128 exhaustive-search engine
//
// Core kernel adapted from burcel/aes-cuda (MIT licence).
// Strategy: encrypt plaintext with every candidate key; compare to ciphertext.
// Tables T0..T4 are the standard AES forward T-tables (same values as the
// reference implementation).
// ============================================================================

#include "aes_cuda.cuh"
#include "logger.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <cstdio>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <string>

// ── Error helpers ─────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess) {                                           \
            char _buf[256];                                                \
            snprintf(_buf, sizeof(_buf), "CUDA error %s:%d – %s",         \
                     __FILE__, __LINE__, cudaGetErrorString(_e));          \
            throw std::runtime_error(_buf);                               \
        }                                                                  \
    } while (0)

// ── Constants (same as reference) ────────────────────────────────────────────
#define TABLE_SIZE   256
#define RCON_SIZE    10
#define U32_SIZE     4
#define MAX_U32      4294967295U
#define BLOCKS       1024
#define THREADS      256          // 256 threads/block – good occupancy on modern GPUs
#define SHARED_MEM_BANK_SIZE 32

// __byte_perm selectors for right-rotate-by-8/16/24
#define SHIFT_1_RIGHT 17185U   // 0x4321
#define SHIFT_2_RIGHT 21554U   // 0x5432
#define SHIFT_3_RIGHT 25923U   // 0x6543

// Round count macros
#define ROUND_COUNT_MIN_1 9

// ── AES forward T-tables (same values as reference, stored in global memory;
//    kernel loads into shared memory) ──────────────────────────────────────────

__device__ __constant__ u32 d_T0[TABLE_SIZE];
__device__ __constant__ u32 d_T4_0[TABLE_SIZE];  // sbox byte 0
__device__ __constant__ u32 d_T4_1[TABLE_SIZE];  // sbox byte 1
__device__ __constant__ u32 d_T4_2[TABLE_SIZE];  // sbox byte 2
__device__ __constant__ u32 d_T4_3[TABLE_SIZE];  // sbox byte 3
__device__ __constant__ u32 d_RCON[RCON_SIZE];

// Host-side table storage (filled by aes_cuda_init)
static u32 h_T0[TABLE_SIZE];
static u32 h_T4_0[TABLE_SIZE], h_T4_1[TABLE_SIZE];
static u32 h_T4_2[TABLE_SIZE], h_T4_3[TABLE_SIZE];
static u32 h_RCON[RCON_SIZE];
static bool g_tables_ready = false;

// ── GF(2^8) helpers (host, used to build tables) ─────────────────────────────
/**
 * @brief Performs multiplication in the Galois Field GF(2^8).
 * 
 * Used for building AES T-tables on the host.
 * 
 * @param a First operand.
 * @param b Second operand.
 * @return The result of (a * b) in GF(2^8).
 */
static u8 h_gmul(u8 a, u8 b) {
    u8 p = 0;
    while (b) {
        if (b & 1) p ^= a;
        if (a & 0x80) a = (a << 1) ^ 0x1b;
        else          a <<= 1;
        b >>= 1;
    }
    return p;
}

// AES S-box (standard values)
static const u8 SBOX[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
};

/**
 * @brief Precomputes AES T-tables and RCON on the host.
 */
static void build_tables() {
    // T0[i] = (s2<<24)|(s<<16)|(s<<8)|s3   where s=SBOX[i], s2=gmul(2,s), s3=gmul(3,s)
    for (int i = 0; i < 256; i++) {
        u8 s  = SBOX[i];
        u8 s2 = h_gmul(2, s);
        u8 s3 = h_gmul(3, s);
        h_T0[i] = ((u32)s2<<24)|((u32)s<<16)|((u32)s<<8)|(u32)s3;

        h_T4_0[i] = (u32)s;
        h_T4_1[i] = (u32)s << 8;
        h_T4_2[i] = (u32)s << 16;
        h_T4_3[i] = (u32)s << 24;
    }
    // RCON
    u8 rc = 1;
    for (int i = 0; i < 10; i++) {
        h_RCON[i] = (u32)rc << 24;
        rc = h_gmul(2, rc);
    }
}

// ── Device helper ─────────────────────────────────────────────────────────────
/**
 * @brief Wrapper for the __byte_perm intrinsic.
 * 
 * @param x Input 32-bit word.
 * @param sel Selector for byte permutation.
 * @return Permuted 32-bit word.
 */
__device__ __forceinline__ u32 ars_byte_perm(u32 x, u32 sel) {
    return __byte_perm(x, x, sel);
}

// ── Main kernel ───────────────────────────────────────────────────────────────
/**
 * @brief Standard CUDA AES-128 brute-force kernel.
 * 
 * Each thread tests a range of consecutive keys.
 * 
 * @param pt Plaintext block (4x32-bit words).
 * @param ct Target ciphertext block (4x32-bit words).
 * @param rk0_fix Upper 32 bits of the 128-bit key (fixed for this launch).
 * @param rk1_fix Second 32 bits of the 128-bit key (fixed for this launch).
 * @param rk2_base Starting value for the third 32-bit word of the key.
 * @param rk3_base Starting value for the fourth 32-bit word of the key.
 * @param range Number of keys to test per thread.
 * @param d_found Device pointer to an integer flag (1 if found).
 * @param d_found_key Device pointer to store the found key.
 */
__global__ void aes128_bruteforce_kernel(
    const u32* __restrict__ pt,          // plaintext  [4]
    const u32* __restrict__ ct,          // ciphertext [4]
    u32 rk0_fix, u32 rk1_fix,           // upper 64 bits of key (fixed)
    u32 rk2_base, u32 rk3_base,         // lower 64 bits start
    u32 range,                           // keys per thread
    int* d_found,                        // output flag
    u32* d_found_key                     // output key [4]
) {
    int  gidx          = blockIdx.x * blockDim.x + threadIdx.x;
    int  warpIdx       = threadIdx.x & 31;

    // ── Load shared memory ────────────────────────────────────────────────
    __shared__ u32 t0S [TABLE_SIZE][SHARED_MEM_BANK_SIZE];
    __shared__ u32 t4_0S[TABLE_SIZE];
    __shared__ u32 t4_1S[TABLE_SIZE];
    __shared__ u32 t4_2S[TABLE_SIZE];
    __shared__ u32 t4_3S[TABLE_SIZE];
    __shared__ u32 rconS[RCON_SIZE];
    __shared__ u32 ctS  [U32_SIZE];

    if (threadIdx.x < TABLE_SIZE) {
        t4_0S[threadIdx.x] = d_T4_0[threadIdx.x];
        t4_1S[threadIdx.x] = d_T4_1[threadIdx.x];
        t4_2S[threadIdx.x] = d_T4_2[threadIdx.x];
        t4_3S[threadIdx.x] = d_T4_3[threadIdx.x];
        for (int b = 0; b < SHARED_MEM_BANK_SIZE; b++)
            t0S[threadIdx.x][b] = d_T0[threadIdx.x];
        if (threadIdx.x < RCON_SIZE)  rconS[threadIdx.x] = d_RCON[threadIdx.x];
        if (threadIdx.x < U32_SIZE)   ctS  [threadIdx.x] = ct[threadIdx.x];
    }
    __syncthreads();

    // ── Starting key for this thread ──────────────────────────────────────
    u32 rk2Init = rk2_base;
    u32 rk3Init = rk3_base;

    u64 offset = (u64)gidx * range;
    // add offset into the 64-bit counter rk2:rk3
    u64 lo64 = ((u64)rk2Init << 32) | (u64)rk3Init;
    lo64 += offset;
    rk2Init = (u32)(lo64 >> 32);
    rk3Init = (u32)(lo64 & 0xFFFFFFFFULL);

    u32 pt0 = pt[0], pt1 = pt[1], pt2 = pt[2], pt3 = pt[3];

    for (u32 iter = 0; iter < range; iter++) {
        // Early exit if another thread already found the key
        if (*d_found) return;

        u32 rk0 = rk0_fix;
        u32 rk1 = rk1_fix;
        u32 rk2 = rk2Init;
        u32 rk3 = rk3Init;

        u32 s0 = pt0 ^ rk0;
        u32 s1 = pt1 ^ rk1;
        u32 s2 = pt2 ^ rk2;
        u32 s3 = pt3 ^ rk3;

        u32 t0, t1, t2, t3;

        // Rounds 1..9 (key-expansion + T-table AES round)
        #pragma unroll
        for (u8 r = 0; r < ROUND_COUNT_MIN_1; r++) {
            u32 temp = rk3;
            rk0 ^= t4_3S[(temp>>16)&0xff] ^ t4_2S[(temp>>8)&0xff]
                ^  t4_1S[(temp)   &0xff] ^ t4_0S[(temp>>24)]
                ^  rconS[r];
            rk1 ^= rk0;
            rk2 ^= rk1;
            rk3 ^= rk2;

            t0 = t0S[s0>>24][warpIdx]
               ^ ars_byte_perm(t0S[(s1>>16)&0xFF][warpIdx], SHIFT_1_RIGHT)
               ^ ars_byte_perm(t0S[(s2>> 8)&0xFF][warpIdx], SHIFT_2_RIGHT)
               ^ ars_byte_perm(t0S[ s3     &0xFF][warpIdx], SHIFT_3_RIGHT)
               ^ rk0;
            t1 = t0S[s1>>24][warpIdx]
               ^ ars_byte_perm(t0S[(s2>>16)&0xFF][warpIdx], SHIFT_1_RIGHT)
               ^ ars_byte_perm(t0S[(s3>> 8)&0xFF][warpIdx], SHIFT_2_RIGHT)
               ^ ars_byte_perm(t0S[ s0     &0xFF][warpIdx], SHIFT_3_RIGHT)
               ^ rk1;
            t2 = t0S[s2>>24][warpIdx]
               ^ ars_byte_perm(t0S[(s3>>16)&0xFF][warpIdx], SHIFT_1_RIGHT)
               ^ ars_byte_perm(t0S[(s0>> 8)&0xFF][warpIdx], SHIFT_2_RIGHT)
               ^ ars_byte_perm(t0S[ s1     &0xFF][warpIdx], SHIFT_3_RIGHT)
               ^ rk2;
            t3 = t0S[s3>>24][warpIdx]
               ^ ars_byte_perm(t0S[(s0>>16)&0xFF][warpIdx], SHIFT_1_RIGHT)
               ^ ars_byte_perm(t0S[(s1>> 8)&0xFF][warpIdx], SHIFT_2_RIGHT)
               ^ ars_byte_perm(t0S[ s2     &0xFF][warpIdx], SHIFT_3_RIGHT)
               ^ rk3;
            s0=t0; s1=t1; s2=t2; s3=t3;
        }

        // Last round key expansion
        {
            u32 temp = rk3;
            rk0 ^= t4_3S[(temp>>16)&0xff] ^ t4_2S[(temp>>8)&0xff]
                ^  t4_1S[(temp)   &0xff] ^ t4_0S[(temp>>24)]
                ^  rconS[ROUND_COUNT_MIN_1];
        }

        // Final round (SubBytes + ShiftRows, no MixColumns)
        s0 = t4_3S[t0>>24] ^ t4_2S[(t1>>16)&0xff]
           ^ t4_1S[(t2>> 8)&0xff] ^ t4_0S[ t3     &0xFF] ^ rk0;

        if (s0 == ctS[0]) {
            rk1 ^= rk0;
            s1 = t4_3S[t1>>24] ^ t4_2S[(t2>>16)&0xff]
               ^ t4_1S[(t3>> 8)&0xff] ^ t4_0S[ t0     &0xFF] ^ rk1;
            if (s1 == ctS[1]) {
                rk2 ^= rk1;
                s2 = t4_3S[t2>>24] ^ t4_2S[(t3>>16)&0xff]
                   ^ t4_1S[(t0>> 8)&0xff] ^ t4_0S[ t1     &0xFF] ^ rk2;
                if (s2 == ctS[2]) {
                    rk3 ^= rk2;
                    s3 = t4_3S[t3>>24] ^ t4_2S[(t0>>16)&0xff]
                       ^ t4_1S[(t1>> 8)&0xff] ^ t4_0S[ t2     &0xFF] ^ rk3;
                    if (s3 == ctS[3]) {
                        // KEY FOUND
                        if (atomicCAS(d_found, 0, 1) == 0) {
                            d_found_key[0] = rk0_fix;
                            d_found_key[1] = rk1_fix;
                            d_found_key[2] = rk2Init;
                            d_found_key[3] = rk3Init;
                        }
                        return;
                    }
                }
            }
        }

        // Increment 64-bit counter rk2:rk3
        if (rk3Init == 0xFFFFFFFFU) rk2Init++;
        rk3Init++;
    }
}

// ── Host-side GPU brute-force ─────────────────────────────────────────────────
/**
 * @brief Host-side wrapper for standard GPU brute-force search.
 * 
 * @param device_id CUDA device index.
 * @param plaintext Known plaintext.
 * @param ciphertext Target ciphertext.
 * @param key_start Starting key.
 * @param num_keys Number of keys to test.
 * @param found_key Output parameter for the found key.
 * @param stop_flag Atomic flag to signal early termination.
 * @param keys_tried Atomic counter for tracking progress.
 * @return True if key was found, false otherwise.
 */
bool gpu_bruteforce(
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

    // Allocate device buffers
    u32 *d_pt, *d_ct, *d_fkey;
    int *d_found;
    CUDA_CHECK(cudaMalloc(&d_pt,    4*sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_ct,    4*sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_fkey,  4*sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_found, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_found,  0, sizeof(int)));

    // Upload plaintext / ciphertext as u32 words
    u32 hpt[4], hct[4];
    plaintext.toWords(hpt);
    ciphertext.toWords(hct);
    CUDA_CHECK(cudaMemcpy(d_pt, hpt, 4*sizeof(u32), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ct, hct, 4*sizeof(u32), cudaMemcpyHostToDevice));

    // We iterate over the upper 64-bit word (rk0:rk1) in the outer loop
    // and search the full lower 64-bit word (rk2:rk3) per GPU launch.
    // key_start encodes the 128-bit start:
    //   rk0 = key_start.w[0], rk1 = key_start.w[1]
    //   rk2 = key_start.w[2], rk3 = key_start.w[3]

    int totalThreads = BLOCKS * THREADS;
    bool result = false;

    u64 remaining = num_keys;
    u64 lo64_start = ((u64)key_start.w[2] << 32) | key_start.w[3];
    u32 rk0 = key_start.w[0];
    u32 rk1 = key_start.w[1];

    while (remaining > 0 && !stop_flag.load(std::memory_order_relaxed)) {
        // batch = how many keys this kernel launch covers
        u64 batch = remaining;
        if (batch > (u64)totalThreads * 0x10000ULL)
            batch = (u64)totalThreads * 0x10000ULL;  // cap at 64 k keys/thread

        u32 per_thread = (u32)((batch + totalThreads - 1) / totalThreads);
        if (per_thread == 0) per_thread = 1;

        u32 rk2_base = (u32)(lo64_start >> 32);
        u32 rk3_base = (u32)(lo64_start & 0xFFFFFFFFULL);

        aes128_bruteforce_kernel<<<BLOCKS, THREADS>>>(
            d_pt, d_ct,
            rk0, rk1,
            rk2_base, rk3_base,
            per_thread,
            d_found, d_fkey
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        u64 processed = (u64)totalThreads * per_thread;
        if (processed > remaining) processed = remaining;
        keys_tried.fetch_add(processed, std::memory_order_relaxed);
        remaining  -= processed;
        lo64_start += processed;

        // Check for carry into upper 64 bits
        if (lo64_start < processed) {
            // overflow: increment rk0:rk1
            u64 hi64 = ((u64)rk0 << 32) | rk1;
            hi64++;
            rk0 = (u32)(hi64 >> 32);
            rk1 = (u32)(hi64 & 0xFFFFFFFFULL);
        }

        int h_found = 0;
        CUDA_CHECK(cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_found) {
            u32 hkey[4];
            CUDA_CHECK(cudaMemcpy(hkey, d_fkey, 4*sizeof(u32), cudaMemcpyDeviceToHost));
            found_key = Key128(hkey[0], hkey[1], hkey[2], hkey[3]);
            stop_flag.store(true, std::memory_order_relaxed);
            result = true;
            break;
        }
    }

    cudaFree(d_pt); cudaFree(d_ct); cudaFree(d_fkey); cudaFree(d_found);
    return result;
}

// ── Benchmark ─────────────────────────────────────────────────────────────────
/**
 * @brief Benchmarks the standard GPU AES performance.
 * 
 * @param device_id CUDA device index.
 * @param duration_ms Target duration.
 * @return Keys per second.
 */
double gpu_benchmark(int device_id, int duration_ms) {
    CUDA_CHECK(cudaSetDevice(device_id));

    // Zero plaintext / ciphertext
    u32 hpt[4]={0}, hct[4]={0};
    u32 *d_pt,*d_ct,*d_fkey; int *d_found;
    CUDA_CHECK(cudaMalloc(&d_pt,   4*sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_ct,   4*sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_fkey, 4*sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_found,  sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pt, hpt,4*sizeof(u32),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ct, hct,4*sizeof(u32),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_found,0,sizeof(int)));

    int totalThreads = BLOCKS * THREADS;
    u32 per_thread   = 64;  // small batch for timing

    auto t0 = std::chrono::high_resolution_clock::now();
    u64  count = 0;

    while (true) {
        aes128_bruteforce_kernel<<<BLOCKS,THREADS>>>(
            d_pt, d_ct, 0, 0, 0, count, per_thread, d_found, d_fkey);
        cudaDeviceSynchronize();
        count += (u64)totalThreads * per_thread;

        auto now = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(now - t0).count();
        if (ms >= duration_ms) {
            cudaFree(d_pt); cudaFree(d_ct); cudaFree(d_fkey); cudaFree(d_found);
            return count / (ms / 1000.0);
        }
    }
}

// ── Init ──────────────────────────────────────────────────────────────────────
/**
 * @brief Initialises AES T-tables on all available CUDA devices.
 */
void aes_cuda_init() {
    if (g_tables_ready) return;
    build_tables();

    int n = 0;
    cudaGetDeviceCount(&n);
    for (int i = 0; i < n; i++) {
        cudaSetDevice(i);
        CUDA_CHECK(cudaMemcpyToSymbol(d_T0,   h_T0,   sizeof(h_T0)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_T4_0, h_T4_0, sizeof(h_T4_0)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_T4_1, h_T4_1, sizeof(h_T4_1)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_T4_2, h_T4_2, sizeof(h_T4_2)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_T4_3, h_T4_3, sizeof(h_T4_3)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_RCON, h_RCON, sizeof(h_RCON)));
    }
    g_tables_ready = true;
}

/**
 * @brief Returns the number of available CUDA devices.
 * 
 * @return Count of GPUs.
 */
int get_gpu_count() {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
}
