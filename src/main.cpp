// ============================================================================
// main.cpp – Entry point for the hybrid AES-128 brute-force engine
//
// Usage:
//   mpirun -n <N+1> ./aes_bruteforce <options>
//
// Options (hex strings, no "0x" prefix):
//   --pt  <hex16>   Plaintext  (16 bytes = 32 hex chars)
//   --ct  <hex16>   Ciphertext (16 bytes = 32 hex chars)
//   --ks  <hex16>   Key space start (default: 0)
//   --ke  <hex16>   Key space end   (default: all-ones)
//
// Rank 0 acts as Master; ranks 1..N act as Workers.
// ============================================================================

#include "utils.h"
#include "logger.h"
#include "aes_cuda.cuh"
#include "master.h"
#include "worker.h"

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdexcept>

// ── Hex parsing ───────────────────────────────────────────────────────────────
static u8 hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    throw std::invalid_argument(std::string("bad hex char: ") + c);
}

static Block128 parse_block(const char* hex) {
    if (strlen(hex) != 32) throw std::invalid_argument("need 32 hex chars");
    Block128 b;
    for (int i = 0; i < 16; i++)
        b.data[i] = (hex_nibble(hex[2*i]) << 4) | hex_nibble(hex[2*i+1]);
    return b;
}

static Key128 parse_key(const char* hex) {
    if (strlen(hex) != 32) throw std::invalid_argument("need 32 hex chars");
    u8 b[16];
    for (int i = 0; i < 16; i++)
        b[i] = (hex_nibble(hex[2*i]) << 4) | hex_nibble(hex[2*i+1]);
    return Key128::fromBytes(b);
}

// ── Default test vectors (FIPS-197 appendix B) ───────────────────────────────
// Key:        2b7e151628aed2a6abf7158809cf4f3c
// Plaintext:  3243f6a8885a308d313198a2e0370734
// Ciphertext: 3925841d02dc09fbdc118597196a0b32
// Search: use a tiny range around the real key for demonstration.
static const char* DEFAULT_PT = "3243f6a8885a308d313198a2e0370734";
static const char* DEFAULT_CT = "3925841d02dc09fbdc118597196a0b32";
// Start slightly before the real key (last byte = 0x00 instead of 0x3c)
static const char* DEFAULT_KS = "2b7e151628aed2a6abf7158809cf4f00";
// End slightly after                (last byte = 0x40)
static const char* DEFAULT_KE = "2b7e151628aed2a6abf7158809cf4f40";

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            fprintf(stderr, "Need at least 2 MPI ranks (1 master + 1 worker).\n");
        MPI_Finalize();
        return 1;
    }

    Logger::instance().init(rank);

    // ── Parse arguments ───────────────────────────────────────────────────
    const char* pt_hex = DEFAULT_PT;
    const char* ct_hex = DEFAULT_CT;
    const char* ks_hex = DEFAULT_KS;
    const char* ke_hex = DEFAULT_KE;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pt") && i+1 < argc) pt_hex = argv[++i];
        else if (!strcmp(argv[i], "--ct") && i+1 < argc) ct_hex = argv[++i];
        else if (!strcmp(argv[i], "--ks") && i+1 < argc) ks_hex = argv[++i];
        else if (!strcmp(argv[i], "--ke") && i+1 < argc) ke_hex = argv[++i];
    }

    Block128 pt, ct;
    Key128   ks, ke;
    try {
        pt = parse_block(pt_hex);
        ct = parse_block(ct_hex);
        ks = parse_key(ks_hex);
        ke = parse_key(ke_hex);
    } catch (const std::exception& e) {
        if (rank == 0) fprintf(stderr, "Argument error: %s\n", e.what());
        MPI_Finalize();
        return 1;
    }

    // Initialise CUDA tables on all ranks (workers need it)
    try {
        aes_cuda_init();
    } catch (const std::exception& e) {
        LOG_WARN("CUDA init: %s – GPU disabled on this rank.", e.what());
    }

    // ── Dispatch ──────────────────────────────────────────────────────────
    if (rank == 0) {
        // ── MASTER ────────────────────────────────────────────────────────
        int num_workers = size - 1;
        Master master(num_workers);
        WorkResult wr = master.run(pt, ct, ks, ke);

        if (wr.found) {
            printf("\n[RESULT] Key found: %08x%08x%08x%08x\n",
                wr.key[0], wr.key[1], wr.key[2], wr.key[3]);
        } else {
            printf("\n[RESULT] Key not found in the given range.\n");
        }
    } else {
        // ── WORKER ────────────────────────────────────────────────────────
        Worker worker(rank);
        worker.benchmark();

        while (true) {
            WorkAssignment wa;
            MPI_Recv(&wa, sizeof(WorkAssignment), MPI_BYTE,
                     0, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // STOP signal: key_end all zeros
            bool stop = true;
            for (int i=0;i<4;i++) if (wa.key_end[i]) { stop=false; break; }
            if (stop) {
                LOG_INFO("Received STOP signal. Exiting.");
                break;
            }

            LOG_INFO("Received work: [%08x%08x%08x%08x .. %08x%08x%08x%08x)",
                wa.key_start[0],wa.key_start[1],wa.key_start[2],wa.key_start[3],
                wa.key_end[0],  wa.key_end[1],  wa.key_end[2],  wa.key_end[3]);

            WorkResult wr = worker.process(wa);
            MPI_Send(&wr, sizeof(WorkResult), MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);

            if (wr.found) {
                LOG_INFO("Key found! Reported to master.");
                // Wait for STOP
                MPI_Recv(&wa, sizeof(WorkAssignment), MPI_BYTE,
                         0, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                break;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
