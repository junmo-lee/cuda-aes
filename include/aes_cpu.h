#pragma once

#include "utils.h"
#include <atomic>

// AES-NI based brute-force (CPU).
// Returns true if the key is found within [key_start, key_start + num_keys).
bool cpu_bruteforce(
    const Block128&    plaintext,
    const Block128&    ciphertext,
    Key128             key_start,
    u64                num_keys,
    Key128&            found_key,
    std::atomic<bool>& stop_flag,
    std::atomic<u64>&  keys_tried
);

// Returns estimated keys/second using AES-NI.
double cpu_benchmark(int duration_ms = 500);
