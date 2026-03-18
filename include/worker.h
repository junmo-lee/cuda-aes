#pragma once

#include "utils.h"
#include <vector>

// Worker node: manages all GPUs + CPU threads on this MPI rank.
class Worker {
public:
    explicit Worker(int rank);

    // Benchmark all devices and store speeds.
    void benchmark();

    // Process a work assignment, returns a result.
    WorkResult process(const WorkAssignment& wa);

    int rank() const { return rank_; }

private:
    struct DeviceRange {
        enum Type { GPU, CPU } type;
        int   id;        // GPU device id (ignored for CPU)
        Key128 start;
        u64   count;
    };

    std::vector<DeviceRange> split_work(Key128 start, u64 total) const;

    int                 rank_;
    int                 num_gpus_;
    double              cpu_speed_ = 0.0;
    std::vector<double> gpu_speeds_;     // T-table kernel speeds
    std::vector<double> bs_gpu_speeds_;  // bitsliced kernel speeds
};
