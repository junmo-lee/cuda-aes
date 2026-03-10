// ============================================================================
// worker.cpp – Worker node: benchmark, split key space, launch GPU/CPU threads
// ============================================================================

#include "worker.h"
#include "aes_cuda.cuh"
#include "aes_cpu.h"
#include "logger.h"

#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <chrono>
#include <cstring>
#include <algorithm>

Worker::Worker(int rank)
    : rank_(rank), num_gpus_(get_gpu_count()) {}

// ── Benchmark ─────────────────────────────────────────────────────────────────
void Worker::benchmark() {
    LOG_INFO("Benchmarking devices…");

    // GPU
    gpu_speeds_.resize(num_gpus_, 0.0);
    for (int g = 0; g < num_gpus_; g++) {
        gpu_speeds_[g] = gpu_benchmark(g);
        LOG_INFO("  GPU[%d]: %.2e keys/s", g, gpu_speeds_[g]);
    }

    // CPU (use hardware_concurrency - 1, reserve 1 thread for monitoring)
    cpu_speed_ = cpu_benchmark();
    unsigned hw = std::thread::hardware_concurrency();
    unsigned cpu_threads = (hw > 2) ? hw - 1 : 1;  // reserve 1 for monitoring
    cpu_speed_ *= cpu_threads;
    LOG_INFO("  CPU (%u threads): %.2e keys/s", cpu_threads, cpu_speed_);
}

// ── Work splitting ────────────────────────────────────────────────────────────
std::vector<Worker::DeviceRange>
Worker::split_work(Key128 start, u64 total) const {
    // Sum of all speeds
    double total_speed = cpu_speed_;
    for (double s : gpu_speeds_) total_speed += s;

    std::vector<DeviceRange> ranges;

    if (total_speed == 0.0) {
        // Fallback: equal split
        u64 per = total / (num_gpus_ + 1);
        for (int g = 0; g < num_gpus_; g++) {
            ranges.push_back({DeviceRange::GPU, g, start, per});
            start += per;
        }
        ranges.push_back({DeviceRange::CPU, 0, start, total - (u64)num_gpus_ * per});
        return ranges;
    }

    u64 assigned = 0;
    for (int g = 0; g < num_gpus_; g++) {
        u64 count = (u64)((gpu_speeds_[g] / total_speed) * total);
        if (count == 0) count = 1;
        ranges.push_back({DeviceRange::GPU, g, start, count});
        start    += count;
        assigned += count;
    }
    // CPU gets the rest
    u64 cpu_count = (total > assigned) ? (total - assigned) : 0;
    if (cpu_count > 0)
        ranges.push_back({DeviceRange::CPU, 0, start, cpu_count});

    return ranges;
}

// ── Process ───────────────────────────────────────────────────────────────────
WorkResult Worker::process(const WorkAssignment& wa) {
    Block128 pt(wa.plaintext);
    Block128 ct(wa.ciphertext);

    Key128 ks = Key128(wa.key_start[0], wa.key_start[1],
                       wa.key_start[2], wa.key_start[3]);
    Key128 ke = Key128(wa.key_end[0],   wa.key_end[1],
                       wa.key_end[2],   wa.key_end[3]);

    // Compute total keys to search
    // We keep it as a u64 approximation; for ranges > 2^64 this would need
    // extended arithmetic – in practice each node's chunk is far smaller.
    // Simple: use the 64-bit lower word difference as an approximation.
    u64 lo_s = ((u64)ks.w[2]<<32)|ks.w[3];
    u64 lo_e = ((u64)ke.w[2]<<32)|ke.w[3];
    u64 total = lo_e - lo_s;  // works for same upper-64 partition
    if (total == 0) total = ~0ULL; // full lower 64-bit space

    auto ranges = split_work(ks, total);

    std::atomic<bool> stop_flag(false);
    std::atomic<u64>  keys_tried(0);

    // ── Shared result ─────────────────────────────────────────────────────
    std::mutex   result_mtx;
    bool         found   = false;
    Key128       found_key;

    // ── Launch threads ────────────────────────────────────────────────────
    std::vector<std::thread> threads;

    for (auto& dr : ranges) {
        if (dr.type == DeviceRange::GPU) {
            threads.emplace_back([&, dr]() {
                Key128 fk;
                bool ok = gpu_bruteforce(
                    dr.id, pt, ct, dr.start, dr.count,
                    fk, stop_flag, keys_tried);
                if (ok) {
                    std::lock_guard<std::mutex> lk(result_mtx);
                    found     = true;
                    found_key = fk;
                }
            });
        } else {
            unsigned hw = std::thread::hardware_concurrency();
            unsigned cpu_threads = (hw > 2) ? hw - 1 : 1;
            u64 per = dr.count / cpu_threads;
            for (unsigned t = 0; t < cpu_threads; t++) {
                u64 cnt   = (t == cpu_threads-1) ? dr.count - t*per : per;
                Key128 s  = dr.start + t * per;
                threads.emplace_back([&, s, cnt]() {
                    Key128 fk;
                    bool ok = cpu_bruteforce(
                        pt, ct, s, cnt, fk, stop_flag, keys_tried);
                    if (ok) {
                        std::lock_guard<std::mutex> lk(result_mtx);
                        found     = true;
                        found_key = fk;
                    }
                });
            }
        }
    }

    // ── Monitoring / reservation thread ──────────────────────────────────
    // (runs on the main thread by periodically checking progress)
    auto monitor = std::thread([&]() {
        using namespace std::chrono;
        auto t_prev = high_resolution_clock::now();
        u64  prev   = 0;
        while (!stop_flag.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(seconds(5));
            u64 cur   = keys_tried.load(std::memory_order_relaxed);
            auto now  = high_resolution_clock::now();
            double dt = duration<double>(now - t_prev).count();
            double kps = (dt > 0) ? (cur - prev) / dt : 0;
            LOG_INFO("Progress: %.2e keys tried, speed=%.2e keys/s",
                     (double)cur, kps);
            prev   = cur;
            t_prev = now;
        }
    });

    for (auto& t : threads) t.join();
    stop_flag.store(true); // stop monitor
    monitor.join();

    WorkResult wr;
    wr.rank  = rank_;
    wr.found = found ? 1 : 0;
    if (found) {
        wr.key[0]=found_key.w[0]; wr.key[1]=found_key.w[1];
        wr.key[2]=found_key.w[2]; wr.key[3]=found_key.w[3];
    } else {
        memset(wr.key, 0, sizeof(wr.key));
    }
    return wr;
}
