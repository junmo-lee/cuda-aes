#pragma once

#include "utils.h"
#include <atomic>

/**
 * @brief Performs AES brute-force search using a bitsliced GPU implementation.
 * 
 * This function utilizes a bitsliced AES-128 kernel that processes multiple 
 * candidate keys per thread (e.g., 32 keys using 32-bit registers) for 
 * improved throughput on the GPU.
 * 
 * @param device_id The CUDA device ID to use for the search.
 * @param plaintext The 128-bit known plaintext.
 * @param ciphertext The 128-bit target ciphertext.
 * @param key_start The inclusive starting key of the search range.
 * @param num_keys The total number of keys to check in this work unit.
 * @param found_key [out] The key found during the search (valid only if return is true).
 * @param stop_flag Atomic flag to signal early termination of the search.
 * @param keys_tried [out] Atomic counter for the number of keys processed.
 * @return true if the key is found within the range, false otherwise.
 * 
 * @note This implementation tests 32 candidate keys per GPU thread.
 */
bool gpu_bruteforce_bs(
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
 * @brief Benchmarks the bitsliced GPU AES-128 performance.
 * 
 * @param device_id The CUDA device ID to benchmark.
 * @param duration_ms The duration of the benchmark in milliseconds.
 * @return Estimated number of keys checked per second for the given device.
 */
double gpu_benchmark_bs(int device_id, int duration_ms = 500);
