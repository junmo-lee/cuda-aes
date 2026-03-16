# CUDA AES-128 Bitslice (32-way lanes) — `TMP` + `constexpr` secret keys

This repo provides:
- A 32-way **bitsliced AES-128** implementation in CUDA
- **Compile-time round keys** (no runtime key storage) using templates/`constexpr`
- **Implicit ShiftRows** (via column remapping) in the main rounds
- Tools for **input generation**, **unpacking**, and **verification**

## Build

```bash
cmake -S . -B build -DGPU_ARCH=90   # adjust SM (e.g. 86, 89)
cmake --build build -j
```

> Build prints ptxas info; expect `0 bytes lmem` for kernels.

## Generate inputs (control grid & block)

```bash
python3 tools/make_inputs.py --grid 2x1x1 --block 256x1x1 --seed 1
# writes inputs/run_YYYYmmdd_HHMMSS/{plaintexts.bin, plaintexts.hex, slices_u32_le.bin, meta.json}
```

- `groups = grid.x*grid.y*grid.z * block.x*block.y*block.z`
- Each thread processes **32** plaintexts (one 128-bit state in bitslice).
- Bitsliced input layout: `groups * 128` little-endian `uint32_t` slices.

## Run (full AES)

```bash
IN=inputs/run_*/slices_u32_le.bin
OUT=outputs/run_full_slices_u32_le.bin
./build/cuda-aes-full "$IN" "$OUT" 2x1x1 256x1x1
```

- Stores **bitsliced ciphertext** to `OUT`.

## Verify and unpack to standard bytes

```bash
python3 tools/verify_outputs.py \
  --meta inputs/run_*/meta.json \
  --slices_out "$OUT" \
  --keyhex 2b7e151628aed2a6abf7158809cf4f3c
```

- Also writes:
  - `outputs/ciphertexts_from_cuda.bin` (unpacked CUDA output)
  - `outputs/ciphertexts_from_python.bin` (Python AES-128 reference)

## Other test targets

- S-box only:
  ```bash
  ./build/cuda-aes-sbox-only inputs/.../slices_u32_le.bin outputs/sbox_only.bin 256
  ```
- MixColumns only:
  ```bash
  ./build/cuda-aes-mix-only inputs/.../slices_u32_le.bin outputs/mix_only.bin 256
  ```

## Compile-time keys (no runtime storage)

- Round keys are computed in templates (`include/aes_keys.hpp`).  
- AddRoundKey is emitted as compile-time `~reg` for key-bit=1 (`xor` with all-ones), using **no** registers or memory for keys.
- Edit the key in `src/aes_full_kernel.cu` inside `run_aes_bs_full()`:
  ```cpp
  using MyKey = StaticKey<
      0x2B,0x7E,0x15,0x16, 0x28,0xAE,0xD2,0xA6, 0xAB,0xF7,0x15,0x88, 0x09,0xCF,0x4F,0x3C
  >;
  ```

## Bitsliced Brute-Force (`cuda-aes-bruteforce-bs`)

Unlike the fixed-key encryption targets, this executable performs an **exhaustive key search** using a fully runtime bitsliced AES-128 implementation.

### How it works

| Feature | Detail |
|---|---|
| Keys per thread | **32** (one per bitsliced lane) |
| Key schedule | Computed at runtime via `bs_keyschedule.cuh` |
| Key generation | 32 consecutive 128-bit keys per thread (`base + lane`) |
| Match detection | 128-bit comparison across all lanes with a single bitmask |
| Found key reporting | `atomicCAS` – first matching lane wins |

### Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cuda-aes-bruteforce-bs -j
```

### Run

```bash
./build/cuda-aes-bruteforce-bs \
    --pt  <32hex>   \   # known plaintext
    --ct  <32hex>   \   # target ciphertext
    --ks  <32hex>   \   # key range start (inclusive)
    --ke  <32hex>   \   # key range end   (exclusive)
    [--blocks  N]   \   # CUDA grid x-dim  (default: auto)
    [--threads N]       # threads per block (default: 256)
```

**FIPS-197 test vector** (key = `2b7e151628aed2a6abf7158809cf4f3c`):

```bash
./build/cuda-aes-bruteforce-bs \
    --pt 3243f6a8885a308d313198a2e0370734 \
    --ct 3925841d02dc09fbdc118597196a0b32 \
    --ks 2b7e151628aed2a6abf7158809cf4f00 \
    --ke 2b7e151628aed2a6abf7158809cf4f40
# → KEY FOUND: 2b7e151628aed2a6abf7158809cf4f3c
```

### New source files

| File | Purpose |
|---|---|
| `include/bs_keyschedule.cuh` | Runtime bitsliced key schedule (`bs_ks_init_from_counter`, `bs_ks_expand_inplace`) |
| `src/aes_bruteforce_bs_kernel.cu` | `aes128_bs32_bruteforce` kernel + `run_aes_bs_bruteforce` host launcher |
| `src/run_bruteforce_bs.cu` | `main()` — CLI argument parsing and kernel dispatch |

### Performance

Measured on T4 GPU (sm_75):

- **~4 Gkeys/s** with a full grid (4096 × 256 threads)
- Register pressure: 255 registers used, ~856 bytes local-memory spill (unavoidable with runtime key schedule)

## Notes
- ShiftRows in the **main rounds** is handled implicitly by calling MixColumns with bytes `{0,5,10,15}`, `{4,9,14,3}`, `{8,13,2,7}`, `{12,1,6,11}`.
- Final round does SubBytes + ShiftRows only, then applies the last round key. A register-only permutation is used once.
