#pragma once

#include "utils.h"
#include <atomic>

/**
 * @brief Performs AES brute-force search using CPU AES-NI instructions.
 * 
 * This function searches for a key that encrypts the given plaintext into the 
 * provided ciphertext within a specified key range.
 * 
 * @param plaintext The 128-bit known plaintext.
 * @param ciphertext The 128-bit target ciphertext.
 * @param key_start The inclusive starting key of the search range.
 * @param num_keys The number of keys to check starting from key_start.
 * @param found_key [out] The key found during the search (only valid if return is true).
 * @param stop_flag Atomic flag to signal early termination of the search.
 * @param keys_tried [out] Atomic counter for the number of keys processed.
 * @return true if the key is found within the range, false otherwise.
 */
bool cpu_bruteforce(
    const Block128&    plaintext,
    const Block128&    ciphertext,
    Key128             key_start,
    u64                num_keys,
    Key128&            found_key,
    std::atomic<bool>& stop_flag,
    std::atomic<u64>&  keys_tried
);

/**
 * @brief Benchmarks the CPU's AES-NI performance.
 * 
 * @param duration_ms The duration of the benchmark in milliseconds.
 * @return Estimated number of keys checked per second.
 */
double cpu_benchmark(int duration_ms = 500);
