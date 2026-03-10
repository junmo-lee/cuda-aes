# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
# Configure (default CUDA architectures: 70;80;86;90)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Optional: specify CUDA architectures
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;90"

# Build
cmake --build build -j$(nproc)
```

## Run

```bash
# Local test (minimum 2 MPI ranks: 1 master + 1 worker)
mpirun --allow-run-as-root -n 2 ./build/aes_bruteforce \
    --pt 3243f6a8885a308d313198a2e0370734 \
    --ct 3925841d02dc09fbdc118597196a0b32 \
    --ks 2b7e151628aed2a6abf7158809cf4f00 \
    --ke 2b7e151628aed2a6abf7158809cf4f40

# HPC cluster (SLURM)
sbatch submit.sh
tail -f logs/node_0.log
```

**FIPS-197 test vector:** Key=`2b7e151628aed2a6abf7158809cf4f3c`, PT=`3243f6a8885a308d313198a2e0370734`, CT=`3925841d02dc09fbdc118597196a0b32`

## Architecture

**Master-Worker MPI pattern** — rank 0 is master, ranks 1..N are workers.

**Master (`master.cpp`):** Partitions the 128-bit key space and dispatches `WorkAssignment` structs to workers via MPI. Waits for `WorkResult` replies; terminates all workers on first match.

**Worker (`worker.cpp`):** On startup, benchmarks each GPU (via `aes_cuda.cu`) and CPU threads (via `aes_cpu.cpp`), then splits the received key range proportionally (e.g., 49%/49%/2% for dual-GPU+CPU). Launches one thread per GPU (`cudaSetDevice`) plus OpenMP threads for CPU. An atomic stop flag enables early exit across all threads when a key is found.

**GPU kernel (`aes_cuda.cu`, 430 lines):** Each CUDA thread tries a contiguous range of keys. AES round keys are precomputed once and stored in shared memory. Core AES-128 is implemented in CUDA using byte-substitution (SubBytes), ShiftRows, MixColumns, and AddRoundKey.

**CPU path (`aes_cpu.cpp`):** Uses Intel AES-NI intrinsics (`_mm_aesenc_si128`, etc.) with OpenMP parallelism. Requires `-maes -msse4.1` at compile time — CPU without AES-NI support will fail to compile/run.

**Key data structures** (`include/utils.h`):
- `Key128`: 4×u32 big-endian; supports `+=`, `<`, comparison for iteration
- `Block128`: 16×u8 plaintext/ciphertext
- `WorkAssignment` / `WorkResult`: MPI message payloads
- `ProgressReport`: Keys tried + current speed

**Logging** (`logger.cpp`): Singleton, per-rank file `logs/node_<rank>.log`, thread-safe with mutex.

## Known Limitations

- AES-128 ECB only (CTR mode and AES-256 are planned future work)
- Key space split operates on the lower 64 bits (full 128-bit iteration is a future improvement)
- Blocking MPI communication (async improvements possible)
- No fault tolerance for node failures
