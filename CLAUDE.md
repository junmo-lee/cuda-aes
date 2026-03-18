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

**Worker (`worker.cpp`):** On startup, benchmarks each GPU with **both** the T-table kernel (`gpu_benchmark`) and the bitsliced kernel (`gpu_benchmark_bs`), then logs a comparison ratio. Work is split proportionally using bitsliced GPU speeds (the kernel used for actual searching). Launches one thread per GPU running `gpu_bruteforce_bs` plus OpenMP threads for CPU. An atomic stop flag enables early exit across all threads when a key is found.

**GPU kernels — two implementations (benchmarked on every startup):**

| File | Kernel | Strategy | Keys/thread |
|---|---|---|---|
| `aes_cuda.cu` | `aes128_bruteforce_kernel` | T-table lookup in shared memory | 1 (inner loop) |
| `aes_cuda_bs.cu` | `aes128_bs32_bruteforce_kernel` | Bitsliced GF(2) logic, no tables | 32 (lane-parallel) |

Workers benchmark both kernels per GPU at startup and log the comparison ratio. The bitsliced kernel (`gpu_bruteforce_bs`) is used for actual brute-force; the T-table kernel is retained for benchmarking only.

**`aes_cuda.cu` (T-table kernel):** Each CUDA thread tries a contiguous range of keys. T0 table (256×32 shared-memory banks) is loaded into shared memory; `__byte_perm` handles byte rotation. AES-128 implemented via SubBytes/ShiftRows/MixColumns/AddRoundKey.

**`aes_cuda_bs.cu` (bitsliced kernel):** Each CUDA thread tests 32 keys in parallel. State is stored bitsliced: `r[byte*8+bit]` is a `uint32_t` where bit j = bit `bit` of byte `byte` for lane j. S-box is implemented as pure combinatorial GF(2) logic (no lookup tables). Key schedule expanded in-place with `bs_ks_expand_inplace`. Headers sourced from `BitSlice_AES_CUDA_REF/include/`.

**CPU path (`aes_cpu.cpp`):** Uses Intel AES-NI intrinsics (`_mm_aesenc_si128`, etc.) with OpenMP parallelism. Requires `-maes -msse4.1` at compile time — CPU without AES-NI support will fail to compile/run.

**New headers (`include/`):**
- `aes_cuda_bs.cuh`: declares `gpu_bruteforce_bs(device_id, pt, ct, key_start, num_keys, found_key, stop_flag, keys_tried)` and `gpu_benchmark_bs(device_id, duration_ms)`

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
| `include/bs_sbox.cuh` | Bitsliced AES S-box (pure GF(2) combinatorial logic) |
| `include/bs_mixcol.cuh` | Bitsliced MixColumns (`bs_mixcl`, `apply_mixcol<B0,B1,B2,B3>`) |
| `include/bs_helpers.cuh` | Template helpers (`load128`, `store128`, `apply_sbox_bytes`) |
| `src/aes_bruteforce_bs_kernel.cu` | `aes128_bs32_bruteforce` kernel (32 keys/thread) + `run_aes_bs_bruteforce` host launcher |
| `src/run_bruteforce_bs.cu` | CLI driver (`--pt`, `--ct`, `--ks`, `--ke`, `--blocks`, `--threads`) |

> These headers are also consumed by the main project via `BitSlice_AES_CUDA_REF/include/` in the top-level `CMakeLists.txt`.

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
