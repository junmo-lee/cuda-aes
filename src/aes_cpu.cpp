// ============================================================================
// aes_cpu.cpp – AES-128 encryption using Intel AES-NI intrinsics (wmmintrin.h)
//
// Strategy mirrors the GPU: encrypt plaintext with candidate key, compare
// with target ciphertext.
// ============================================================================

#include "aes_cpu.h"

#include <wmmintrin.h>   // _mm_aesenc_si128 etc.
#include <emmintrin.h>   // _mm_loadu_si128 etc.
#include <immintrin.h>

#include <chrono>
#include <cstring>

// ── AES-NI key expansion ──────────────────────────────────────────────────────
/**
 * @brief Expands one round key using the result of the AESKEYGENASSIST instruction.
 * 
 * Performs the necessary shuffle and XOR operations to generate the next 128-bit
 * round key from the previous one and the generated assistance value.
 * 
 * @param key The previous round key.
 * @param kgen The result of _mm_aeskeygenassist_si128.
 * @return The expanded 128-bit round key.
 */
static __m128i expand_key_step(__m128i key, __m128i kgen) {
    kgen = _mm_shuffle_epi32(kgen, 0xFF);
    key  = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key  = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key  = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    return _mm_xor_si128(key, kgen);
}

/**
 * @brief Expands a 128-bit key into 11 round keys for AES-128 encryption.
 * 
 * Uses AES-NI intrinsics to perform the standard AES-128 key schedule.
 * 
 * @param key The 16-byte (128-bit) input key.
 * @param rk Array of 11 __m128i elements to store the expanded round keys.
 */
static void aes128_key_schedule(const u8 key[16], __m128i rk[11]) {
    rk[0]  = _mm_loadu_si128((__m128i*)key);
    rk[1]  = expand_key_step(rk[0],  _mm_aeskeygenassist_si128(rk[0],  0x01));
    rk[2]  = expand_key_step(rk[1],  _mm_aeskeygenassist_si128(rk[1],  0x02));
    rk[3]  = expand_key_step(rk[2],  _mm_aeskeygenassist_si128(rk[2],  0x04));
    rk[4]  = expand_key_step(rk[3],  _mm_aeskeygenassist_si128(rk[3],  0x08));
    rk[5]  = expand_key_step(rk[4],  _mm_aeskeygenassist_si128(rk[4],  0x10));
    rk[6]  = expand_key_step(rk[5],  _mm_aeskeygenassist_si128(rk[5],  0x20));
    rk[7]  = expand_key_step(rk[6],  _mm_aeskeygenassist_si128(rk[6],  0x40));
    rk[8]  = expand_key_step(rk[7],  _mm_aeskeygenassist_si128(rk[7],  0x80));
    rk[9]  = expand_key_step(rk[8],  _mm_aeskeygenassist_si128(rk[8],  0x1b));
    rk[10] = expand_key_step(rk[9],  _mm_aeskeygenassist_si128(rk[9],  0x36));
}

/**
 * @brief Encrypts a single 128-bit block using AES-128.
 * 
 * Employs AES-NI intrinsics for high-performance encryption.
 * 
 * @param block The 128-bit block (plaintext) to encrypt.
 * @param rk The 11 expanded round keys.
 * @return The 128-bit encrypted block (ciphertext).
 */
static __m128i aes128_encrypt(__m128i block, const __m128i rk[11]) {
    block = _mm_xor_si128(block, rk[0]);
    block = _mm_aesenc_si128   (block, rk[1]);
    block = _mm_aesenc_si128   (block, rk[2]);
    block = _mm_aesenc_si128   (block, rk[3]);
    block = _mm_aesenc_si128   (block, rk[4]);
    block = _mm_aesenc_si128   (block, rk[5]);
    block = _mm_aesenc_si128   (block, rk[6]);
    block = _mm_aesenc_si128   (block, rk[7]);
    block = _mm_aesenc_si128   (block, rk[8]);
    block = _mm_aesenc_si128   (block, rk[9]);
    block = _mm_aesenclast_si128(block, rk[10]);
    return block;
}

// ── Brute-force ───────────────────────────────────────────────────────────────
/**
 * @brief Performs CPU-based brute-force search for an AES-128 key.
 * 
 * Iterates through a range of keys, encrypting the given plaintext and comparing
 * it to the target ciphertext until the key is found or the range is exhausted.
 * 
 * @param plaintext The known plaintext block.
 * @param ciphertext The target ciphertext block.
 * @param key_start The starting key for the search.
 * @param num_keys The number of keys to test in this range.
 * @param found_key Output parameter to store the key if found.
 * @param stop_flag Atomic flag to signal early termination.
 * @param keys_tried Atomic counter to track the total number of keys processed.
 * @return True if the key was found, false otherwise.
 */
bool cpu_bruteforce(
    const Block128&    plaintext,
    const Block128&    ciphertext,
    Key128             key_start,
    u64                num_keys,
    Key128&            found_key,
    std::atomic<bool>& stop_flag,
    std::atomic<u64>&  keys_tried
) {
    __m128i pt_blk = _mm_loadu_si128((__m128i*)plaintext.data);
    __m128i ct_blk = _mm_loadu_si128((__m128i*)ciphertext.data);

    u8       key_bytes[16];
    __m128i  rk[11];

    Key128 cur = key_start;
    const u64 BATCH = 8192;
    u64 done = 0;

    while (done < num_keys && !stop_flag.load(std::memory_order_relaxed)) {
        u64 batch = (num_keys - done < BATCH) ? (num_keys - done) : BATCH;

        for (u64 i = 0; i < batch; i++) {
            cur.toBytes(key_bytes);
            aes128_key_schedule(key_bytes, rk);
            __m128i enc = aes128_encrypt(pt_blk, rk);

            if (_mm_movemask_epi8(_mm_cmpeq_epi8(enc, ct_blk)) == 0xFFFF) {
                found_key = cur;
                keys_tried.fetch_add(done + i + 1, std::memory_order_relaxed);
                return true;
            }
            cur += 1;
        }

        done += batch;
        keys_tried.fetch_add(batch, std::memory_order_relaxed);
    }
    return false;
}

// ── Benchmark ─────────────────────────────────────────────────────────────────
/**
 * @brief Benchmarks the CPU's AES-128 encryption performance.
 * 
 * Measures how many keys per second the CPU can process using AES-NI.
 * 
 * @param duration_ms The target duration for the benchmark in milliseconds.
 * @return The measured performance in keys per second.
 */
double cpu_benchmark(int duration_ms) {
    u8 key[16] = {};
    u8 pt[16]  = {};
    __m128i pt_blk = _mm_loadu_si128((__m128i*)pt);
    __m128i rk[11];

    auto t0 = std::chrono::high_resolution_clock::now();
    u64 count = 0;

    while (true) {
        for (int i = 0; i < 1024; i++) {
            key[15] = (u8)(count + i);
            aes128_key_schedule(key, rk);
            volatile __m128i r = aes128_encrypt(pt_blk, rk);
            (void)r;
        }
        count += 1024;

        auto now = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(now - t0).count();
        if (ms >= duration_ms)
            return count / (ms / 1000.0);
    }
}
