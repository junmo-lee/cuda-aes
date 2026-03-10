// ============================================================================
// master.cpp – MPI Master: partitions key space and coordinates workers
// ============================================================================

#include "master.h"
#include "logger.h"

#include <mpi.h>
#include <cstring>
#include <vector>
#include <chrono>

Master::Master(int num_workers) : num_workers_(num_workers) {}

// ── MPI send helpers ──────────────────────────────────────────────────────────
void Master::send_work(int dest, const WorkAssignment& wa) {
    MPI_Send(&wa, sizeof(WorkAssignment), MPI_BYTE, dest, TAG_WORK, MPI_COMM_WORLD);
}

void Master::send_stop(int dest) {
    WorkAssignment stop;
    memset(&stop, 0, sizeof(stop));  // all-zero key_end = STOP signal
    MPI_Send(&stop, sizeof(WorkAssignment), MPI_BYTE, dest, TAG_WORK, MPI_COMM_WORLD);
}

WorkResult Master::recv_result(int src) {
    WorkResult wr;
    MPI_Recv(&wr, sizeof(WorkResult), MPI_BYTE, src, TAG_RESULT, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    return wr;
}

// ── Key-space splitter ────────────────────────────────────────────────────────
// Divide [start, end) into `n` roughly equal chunks (operating on lower 64 bits
// with fixed upper 64 bits for simplicity; master iterates upper bits outer).
static void split_range(
    const Key128& start, const Key128& end,
    int n,
    std::vector<std::pair<Key128,Key128>>& chunks
) {
    // Approximate: work on u64 lower half, upper half same for now.
    // For a real full 128-bit split, outer loop over upper 64 bits would be
    // performed in the master run() loop.
    u64 lo_s = ((u64)start.w[2]<<32)|start.w[3];
    u64 lo_e = ((u64)end.w[2]  <<32)|end.w[3];
    u64 total = lo_e - lo_s;
    u64 per   = total / n;
    if (per == 0) per = 1;

    Key128 cur = start;
    for (int i = 0; i < n; i++) {
        u64 cnt  = (i == n-1) ? (lo_e - (((u64)cur.w[2]<<32)|cur.w[3])) : per;
        Key128 nx = cur + cnt;
        chunks.push_back({cur, nx});
        cur = nx;
    }
}

// ── Run ───────────────────────────────────────────────────────────────────────
WorkResult Master::run(
    const Block128& plaintext,
    const Block128& ciphertext,
    Key128          key_start,
    Key128          key_end
) {
    LOG_INFO("Master starting search, %d workers.", num_workers_);
    LOG_INFO("Key start: %s", key_start.toHex().c_str());
    LOG_INFO("Key end  : %s", key_end.toHex().c_str());

    auto t_start = std::chrono::high_resolution_clock::now();

    // Split the key space across workers
    std::vector<std::pair<Key128,Key128>> chunks;
    split_range(key_start, key_end, num_workers_, chunks);

    // Send initial work to all workers (ranks 1..num_workers_)
    for (int w = 0; w < num_workers_; w++) {
        WorkAssignment wa;
        memcpy(wa.plaintext,  plaintext.data,  16);
        memcpy(wa.ciphertext, ciphertext.data, 16);
        for (int j=0;j<4;j++) wa.key_start[j] = chunks[w].first.w[j];
        for (int j=0;j<4;j++) wa.key_end[j]   = chunks[w].second.w[j];

        LOG_INFO("  → worker %d: [%s .. %s)",
            w+1,
            chunks[w].first.toHex().c_str(),
            chunks[w].second.toHex().c_str());
        send_work(w + 1, wa);
    }

    // Collect results
    WorkResult global;
    global.found = 0;
    memset(global.key, 0, sizeof(global.key));

    for (int w = 0; w < num_workers_; w++) {
        WorkResult wr = recv_result(w + 1);
        if (wr.found) {
            global = wr;
            LOG_INFO("!!! KEY FOUND by rank %d: %08x%08x%08x%08x",
                wr.rank, wr.key[0], wr.key[1], wr.key[2], wr.key[3]);
        }
    }

    // Broadcast STOP to all remaining workers
    for (int w = 0; w < num_workers_; w++) {
        send_stop(w + 1);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    if (global.found) {
        LOG_INFO("Search complete in %.1f s. Key: %08x%08x%08x%08x",
            elapsed, global.key[0],global.key[1],global.key[2],global.key[3]);
    } else {
        LOG_INFO("Search complete in %.1f s. Key NOT found in given range.", elapsed);
    }

    return global;
}
