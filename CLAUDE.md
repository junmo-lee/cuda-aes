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

## BitSlice_AES_CUDA_REF — Bitsliced brute-force kernel

A standalone bitsliced AES-128 brute-force lives in `BitSlice_AES_CUDA_REF/`. It is independent of the MPI project and builds its own executable.

### Build

```bash
cd BitSlice_AES_CUDA_REF
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cuda-aes-bruteforce-bs -j$(nproc)
```

### Run

```bash
./build/cuda-aes-bruteforce-bs \
    --pt 3243f6a8885a308d313198a2e0370734 \
    --ct 3925841d02dc09fbdc118597196a0b32 \
    --ks 2b7e151628aed2a6abf7158809cf4f00 \
    --ke 2b7e151628aed2a6abf7158809cf4f40
# → KEY FOUND: 2b7e151628aed2a6abf7158809cf4f3c
```

### Key files added

| File | Role |
|---|---|
| `include/bs_keyschedule.cuh` | Runtime bitsliced key schedule (`bs_ks_init_from_counter`, `bs_ks_expand_inplace`) |
| `src/aes_bruteforce_bs_kernel.cu` | `aes128_bs32_bruteforce` kernel (32 keys/thread) + host launcher |
| `src/run_bruteforce_bs.cu` | CLI driver (`--pt`, `--ct`, `--ks`, `--ke`, `--blocks`, `--threads`) |

### Design notes

- **32 keys per thread**: plaintext is replicated across all 32 bitsliced lanes; each lane uses a distinct key (`base + lane`).
- **Runtime key schedule**: `bs_ks_expand_inplace` runs RotWord → SubWord (via `bs_sbox`) → Rcon XOR → 4-word propagation on bitsliced data.
- **Match detection**: after 10 rounds, a single 128-bit comparison produces a 32-bit lane mask; the winning lane reports via `atomicCAS`.
- **Throughput**: ~4 Gkeys/s on T4 (sm_75), 255 registers used with ~856 B local-memory spill.

## Known Limitations

- AES-128 ECB only (CTR mode and AES-256 are planned future work)
- Key space split operates on the lower 64 bits (full 128-bit iteration is a future improvement)
- Blocking MPI communication (async improvements possible)
- No fault tolerance for node failures
- `cuda-aes-bruteforce-bs`: single kernel launch covers `grid×block×32` keys; callers must loop for ranges larger than that
