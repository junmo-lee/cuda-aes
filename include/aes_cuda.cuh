#pragma once

#include "utils.h"
#include <atomic>

/**
 * @brief Initializes AES lookup tables on the host and potentially the GPU.
 * 
 * This function should be called once per process before any GPU-based 
 * AES operations are performed.
 */
void aes_cuda_init();

/**
 * @brief Performs AES brute-force search using a standard GPU implementation.
 * 
 * Searches for the key in the range [key_start, key_start + num_keys) that 
 * encrypts the given plaintext to the ciphertext.
 * 
 * @param device_id The CUDA device ID to use for the search.
 * @param plaintext The 128-bit known plaintext.
 * @param ciphertext The 128-bit target ciphertext.
 * @param key_start The inclusive starting key of the search range.
 * @param num_keys The total number of keys to check.
 * @param found_key [out] The found key (valid only if return is true).
 * @param stop_flag Atomic flag to signal early termination across all devices.
 * @param keys_tried [out] Atomic counter for the number of keys processed.
 * @return true if the key is found, false otherwise.
 */
bool gpu_bruteforce(
    int                   device_id,
    const Block128&       plaintext,
    const Block128&       ciphertext,
    Key128                key_start,
    u64                   num_keys,
    Key128&               found_key,
    std::atomic<bool>&    stop_flag,
    std::atomic<u64>&     keys_tried
);

/**
 * @brief Benchmarks the standard GPU AES-128 performance.
 * 
 * @param device_id The CUDA device ID to benchmark.
 * @param duration_ms The duration of the benchmark in milliseconds.
 * @return Estimated number of keys checked per second.
 */
double gpu_benchmark(int device_id, int duration_ms = 500);

/**
 * @brief Gets the number of CUDA-capable devices on the current node.
 * @return The number of available CUDA devices.
 */
int get_gpu_count();
