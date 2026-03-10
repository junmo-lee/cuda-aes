#pragma once

#include "utils.h"

// Master node: divides key space, distributes work via MPI, collects results.
class Master {
public:
    explicit Master(int num_workers);

    // Run the search over the full [key_start, key_end) range.
    // Returns a WorkResult with found=1 if a key is found.
    WorkResult run(
        const Block128& plaintext,
        const Block128& ciphertext,
        Key128          key_start,
        Key128          key_end
    );

private:
    void send_work(int dest, const WorkAssignment& wa);
    void send_stop(int dest);
    WorkResult recv_result(int src);

    int num_workers_;
};
