#pragma once

#include "utils.h"
#include <atomic>

// Bitsliced AES-128 GPU brute-force.
// Same interface as gpu_bruteforce (aes_cuda.cuh) but uses the bitsliced
// kernel that tests 32 candidate keys per thread.
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

// Benchmark bitsliced kernel; returns estimated keys/second for device_id.
double gpu_benchmark_bs(int device_id, int duration_ms = 500);
