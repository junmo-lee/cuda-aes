#pragma once

#include "utils.h"

/**
 * @brief Master node class responsible for coordinating the AES brute-force search.
 * 
 * The Master node divides the key space, distributes work assignments to worker 
 * nodes via MPI, and collects results.
 */
class Master {
public:
    /**
     * @brief Constructs a Master object.
     * @param num_workers Total number of worker nodes to coordinate.
     */
    explicit Master(int num_workers);

    /**
     * @brief Initiates and coordinates the full key search.
     * 
     * @param plaintext The 128-bit known plaintext.
     * @param ciphertext The 128-bit target ciphertext.
     * @param key_start The inclusive starting key of the search range.
     * @param key_end The exclusive ending key of the search range.
     * @return A WorkResult object containing the found key or indicating failure.
     */
    WorkResult run(
        const Block128& plaintext,
        const Block128& ciphertext,
        Key128          key_start,
        Key128          key_end
    );

private:
    /**
     * @brief Sends a work assignment to a specific worker node.
     * @param dest The MPI rank of the destination worker node.
     * @param wa The work assignment to send.
     */
    void send_work(int dest, const WorkAssignment& wa);
    
    /**
     * @brief Sends a stop signal to a specific worker node.
     * @param dest The MPI rank of the destination worker node.
     */
    void send_stop(int dest);
    
    /**
     * @brief Receives a work result from a specific worker node.
     * @param src The MPI rank of the source worker node.
     * @return The received WorkResult object.
     */
    WorkResult recv_result(int src);

    int num_workers_; ///< The number of worker nodes participating in the search.
};
