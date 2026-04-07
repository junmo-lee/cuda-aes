// =============================================================================
// run_bruteforce_bs.cu  –  Host driver for bitsliced AES-128 brute-force
//
// Usage:
//   cuda-aes-bruteforce-bs \
//       --pt  <32 hex chars>   plaintext
//       --ct  <32 hex chars>   target ciphertext
//       --ks  <32 hex chars>   key range start (inclusive)
//       --ke  <32 hex chars>   key range end   (exclusive)
//       [--blocks  N]          CUDA grid x-dim  (default: auto)
//       [--threads N]          CUDA block x-dim (default: 256)
//
// FIPS-197 test vector (key = 2b7e...4f3c):
//   --pt 3243f6a8885a308d313198a2e0370734
//   --ct 3925841d02dc09fbdc118597196a0b32
//   --ks 2b7e151628aed2a6abf7158809cf4f00
//   --ke 2b7e151628aed2a6abf7158809cf4f40
// =============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Declared in aes_bruteforce_bs_kernel.cu
extern "C" int run_aes_bs_bruteforce(
        const uint8_t  plaintext[16],
        const uint8_t  ciphertext[16],
        const uint32_t key_start[4],
        const uint32_t key_end[4],
        uint32_t       found_key[4],
        dim3 grid, dim3 block);


// ── Utility helpers ──────────────────────────────────────────────────────────

/**
 * @brief Converts a hexadecimal string to a byte array.
 * 
 * @param hex The input hexadecimal string.
 * @param out [out] The output byte array.
 * @param n The number of bytes to extract.
 * @return true if conversion is successful, false otherwise.
 */
static bool hex_to_bytes(const char* hex, uint8_t* out, int n)
{
    if ((int)strlen(hex) != n * 2) return false;
    for (int i = 0; i < n; i++) {
        auto nibble = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        int hi = nibble(hex[i*2]), lo = nibble(hex[i*2+1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return true;
}

/**
 * @brief Converts a 16-byte array to four 32-bit big-endian words.
 * 
 * @param b The input 16-byte array.
 * @param w [out] The output array of four 32-bit words.
 */
static void bytes_to_words_be(const uint8_t b[16], uint32_t w[4])
{
    for (int i = 0; i < 4; i++)
        w[i] = ((uint32_t)b[i*4+0] << 24) | ((uint32_t)b[i*4+1] << 16)
             | ((uint32_t)b[i*4+2] <<  8) |  (uint32_t)b[i*4+3];
}

/**
 * @brief Prints a label and a byte array in hexadecimal format.
 * 
 * @param label The label to print before the hex string.
 * @param b The byte array to print.
 * @param n The number of bytes to print.
 */
static void print_hex(const char* label, const uint8_t* b, int n)
{
    printf("%-12s", label);
    for (int i = 0; i < n; i++) printf("%02x", b[i]);
    printf("\n");
}


// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    const char* pt_hex  = nullptr;
    const char* ct_hex  = nullptr;
    const char* ks_hex  = nullptr;
    const char* ke_hex  = nullptr;
    int user_blocks  = 0;     // 0 = auto-compute
    int user_threads = 256;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--pt")      && i+1 < argc) pt_hex = argv[++i];
        else if (!strcmp(argv[i], "--ct")      && i+1 < argc) ct_hex = argv[++i];
        else if (!strcmp(argv[i], "--ks")      && i+1 < argc) ks_hex = argv[++i];
        else if (!strcmp(argv[i], "--ke")      && i+1 < argc) ke_hex = argv[++i];
        else if (!strcmp(argv[i], "--blocks")  && i+1 < argc) user_blocks  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i+1 < argc) user_threads = atoi(argv[++i]);
    }

    if (!pt_hex || !ct_hex || !ks_hex || !ke_hex) {
        fprintf(stderr,
            "Usage: %s --pt <32hex> --ct <32hex> --ks <32hex> --ke <32hex>"
            " [--blocks N] [--threads N]\n"
            "\n"
            "FIPS-197 example (key=2b7e...4f3c):\n"
            "  --pt 3243f6a8885a308d313198a2e0370734\n"
            "  --ct 3925841d02dc09fbdc118597196a0b32\n"
            "  --ks 2b7e151628aed2a6abf7158809cf4f00\n"
            "  --ke 2b7e151628aed2a6abf7158809cf4f40\n",
            argv[0]);
        return 1;
    }

    uint8_t pt_b[16], ct_b[16], ks_b[16], ke_b[16];
    if (!hex_to_bytes(pt_hex, pt_b, 16)) { fprintf(stderr, "Bad --pt hex\n"); return 1; }
    if (!hex_to_bytes(ct_hex, ct_b, 16)) { fprintf(stderr, "Bad --ct hex\n"); return 1; }
    if (!hex_to_bytes(ks_hex, ks_b, 16)) { fprintf(stderr, "Bad --ks hex\n"); return 1; }
    if (!hex_to_bytes(ke_hex, ke_b, 16)) { fprintf(stderr, "Bad --ke hex\n"); return 1; }

    uint32_t ks_w[4], ke_w[4];
    bytes_to_words_be(ks_b, ks_w);
    bytes_to_words_be(ke_b, ke_w);

    // Key count (lower 64-bit range).
    const uint64_t ks_lo = ((uint64_t)ks_w[2] << 32) | ks_w[3];
    const uint64_t ke_lo = ((uint64_t)ke_w[2] << 32) | ke_w[3];
    const uint64_t total_keys = ke_lo - ks_lo;

    print_hex("Plaintext:  ", pt_b, 16);
    print_hex("Ciphertext: ", ct_b, 16);
    print_hex("Key start:  ", ks_b, 16);
    print_hex("Key end:    ", ke_b, 16);
    printf("Key range:  %llu keys\n", (unsigned long long)total_keys);

    // Compute grid dimensions.
    const int threads = user_threads;
    int blocks;
    if (user_blocks > 0) {
        blocks = user_blocks;
    } else {
        // One thread per 32 keys, rounded up.
        const uint64_t needed_threads = (total_keys + 31u) / 32u;
        blocks = (int)((needed_threads + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
    }

    printf("Grid:       %d blocks × %d threads = %llu keys/launch\n",
           blocks, threads, (unsigned long long)blocks * threads * 32u);

    dim3 grid(blocks, 1, 1), block(threads, 1, 1);

    uint32_t found_key[4] = {0, 0, 0, 0};
    const int rc = run_aes_bs_bruteforce(pt_b, ct_b, ks_w, ke_w,
                                         found_key, grid, block);

    if (rc == 1) {
        printf("KEY FOUND:  %08x%08x%08x%08x\n",
               found_key[0], found_key[1], found_key[2], found_key[3]);
        return 0;
    } else if (rc == 0) {
        printf("Key not found in [%s, %s).\n", ks_hex, ke_hex);
        return 2;
    } else {
        fprintf(stderr, "Kernel error (rc=%d).\n", rc);
        return 3;
    }
}
