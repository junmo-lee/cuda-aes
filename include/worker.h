#pragma once

#include "utils.h"
#include <vector>

/**
 * @brief Worker node class responsible for performing the brute-force search on a node.
 * 
 * Each worker node manages all available CUDA GPUs and CPU threads on its 
 * respective MPI rank. It receives work assignments from the Master and 
 * executes the search.
 */
class Worker {
public:
    /**
     * @brief Constructs a Worker object.
     * @param rank The MPI rank of this worker node.
     */
    explicit Worker(int rank);

    /**
     * @brief Benchmarks all available computation devices (GPUs and CPU).
     * 
     * Measures the performance of standard GPU, bitsliced GPU, and CPU AES 
     * implementations to inform work distribution.
     */
    void benchmark();

    /**
     * @brief Processes a work assignment from the Master.
     * 
     * Splits the assigned key range among available GPUs and CPU threads, 
     * executes the search, and returns the result.
     * 
     * @param wa The work assignment containing the target ciphertext, 
     * plaintext, and key range.
     * @return A WorkResult object indicating if the key was found.
     */
    WorkResult process(const WorkAssignment& wa);

    /**
     * @brief Gets the MPI rank of this worker.
     * @return The worker's rank.
     */
    int rank() const { return rank_; }

private:
    /**
     * @brief Represents a range of keys assigned to a specific computation device.
     */
    struct DeviceRange {
        /**
         * @brief Enumeration of supported device types.
         */
        enum Type { 
            GPU, ///< CUDA-capable GPU device.
            CPU  ///< CPU cores using AES-NI instructions.
        } type;
        
        int   id;        ///< CUDA device ID (ignored for CPU).
        Key128 start;    ///< Inclusive start of the key range for this device.
        u64   count;     ///< Number of keys for this device to check.
    };

    /**
     * @brief Splits a total key range among available computation devices.
     * 
     * Distributes the work based on the measured benchmarks for each device 
     * to achieve load balancing.
     * 
     * @param start The inclusive starting key of the total range.
     * @param total The total number of keys in the range.
     * @return A vector of DeviceRange structures detailing assignments.
     */
    std::vector<DeviceRange> split_work(Key128 start, u64 total) const;

    int                 rank_;             ///< MPI rank of this worker node.
    int                 num_gpus_;         ///< Number of available GPUs on this node.
    double              cpu_speed_ = 0.0;  ///< Measured speed of the CPU implementation.
    std::vector<double> gpu_speeds_;       ///< Speeds of the standard GPU kernels.
    std::vector<double> bs_gpu_speeds_;    ///< Speeds of the bitsliced GPU kernels.
};
