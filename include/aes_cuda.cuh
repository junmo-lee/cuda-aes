#pragma once

#include "utils.h"
#include <atomic>

// Initialise lookup tables (call once per process before any GPU work).
void aes_cuda_init();

// Brute-force search over [key_start, key_start + num_keys) on one GPU.
// stop_flag : shared flag – set to true when any device finds the key.
// keys_tried: atomic progress counter (keys_tried += processed keys on exit).
// Returns true and sets found_key when a matching key is discovered.
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

// Benchmark: returns estimated keys/second for device_id.
double gpu_benchmark(int device_id, int duration_ms = 500);

// Number of CUDA devices on this node.
int get_gpu_count();
