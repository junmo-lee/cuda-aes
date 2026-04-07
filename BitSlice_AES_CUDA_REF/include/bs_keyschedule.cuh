#pragma once
#include <stdint.h>
#include "bs_sbox.cuh"

/**
 * @brief AES-128 round constants (Rcon[1..10] used by key schedule).
 * Kept as a device-local inline table to avoid constexpr linkage issues.
 */
__device__ __constant__ uint8_t BS_KS_RCON[11] = {
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36
};

// =============================================================================
// Bitsliced AES-128 runtime key schedule
//
// Data layout: rk[byte * 8 + bit] = uint32_t where bit j (0..31) is
//              bit <bit> of byte <byte> of round-key for lane j.
// This mirrors the state layout used in the rest of the bitslice library.
// =============================================================================


/**
 * @brief Build bitsliced round-key-0 from 32 consecutive 128-bit keys.
 * 
 * Lane j encrypts with key = base + j  (128-bit big-endian addition).
 * 
 * @param rk [out] 128 bitsliced slices (round key 0).
 * @param base [in] 128-bit base key as 4 big-endian 32-bit words (base[0] = MSW, base[3] = LSW).
 * @return void
 */
__device__ __forceinline__ void bs_ks_init_from_counter(
        uint32_t rk[128],
        const uint32_t base[4])
{
    for (int i = 0; i < 128; i++) rk[i] = 0;

    for (int j = 0; j < 32; j++) {
        // key_j = base + j  (128-bit, big-endian)
        uint64_t lo = ((uint64_t)base[2] << 32) | base[3];
        uint64_t hi = ((uint64_t)base[0] << 32) | base[1];
        lo += (uint64_t)j;
        if (lo < (uint64_t)j) hi++;

        const uint32_t words[4] = {
            (uint32_t)(hi >> 32),
            (uint32_t)(hi & 0xFFFFFFFFu),
            (uint32_t)(lo >> 32),
            (uint32_t)(lo & 0xFFFFFFFFu)
        };

        // Pack each bit of each key byte into the corresponding slice.
        // Key byte k lives in word k/4.  Within that word (big-endian)
        // the shift to reach byte k%4 is 8*(3 - k%4).
        for (int k = 0; k < 16; k++) {
            const int     wi    = k / 4;
            const int     shift = 8 * (3 - (k % 4));
            const uint8_t bval  = (words[wi] >> shift) & 0xFFu;
            for (int b = 0; b < 8; b++) {
                if ((bval >> b) & 1u)
                    rk[k * 8 + b] |= (1u << j);
            }
        }
    }
}


/**
 * @brief Expand bitsliced round key in-place: rk[] = RK_{rnd-1} -> RK_{rnd}.
 * 
 * rnd must be in [1, 10].
 * 
 * AES-128 key schedule (per round):
 *   temp    = SubWord(RotWord(W3)) XOR Rcon[rnd]
 *   new_W0  = W0 XOR temp
 *   new_W1  = W1 XOR new_W0
 *   new_W2  = W2 XOR new_W1
 *   new_W3  = W3 XOR new_W2
 * 
 * In the bitsliced layout W0 occupies rk[0..31], W1 rk[32..63],
 * W2 rk[64..95], W3 rk[96..127].
 * 
 * @param rk [in,out] The 128-slice bitsliced round key state.
 * @param rnd [in] The current round number (1-10).
 * @return void
 */
__device__ __forceinline__ void bs_ks_expand_inplace(uint32_t rk[128], int rnd)
{
    // ── Step 1: RotWord(W3) ─────────────────────────────────────────────
    // W3 = bytes 12,13,14,15.
    // RotWord({b12,b13,b14,b15}) = {b13,b14,b15,b12}
    uint32_t rot[32];
    for (int b = 0; b < 8; b++) {
        rot[ 0+b] = rk[13*8+b];   // rot byte 0  ← old byte 13
        rot[ 8+b] = rk[14*8+b];   // rot byte 1  ← old byte 14
        rot[16+b] = rk[15*8+b];   // rot byte 2  ← old byte 15
        rot[24+b] = rk[12*8+b];   // rot byte 3  ← old byte 12
    }

    // ── Step 2: SubWord – S-box each of the 4 bytes ─────────────────────
    bs_sbox(rot[ 0], rot[ 1], rot[ 2], rot[ 3],
            rot[ 4], rot[ 5], rot[ 6], rot[ 7]);
    bs_sbox(rot[ 8], rot[ 9], rot[10], rot[11],
            rot[12], rot[13], rot[14], rot[15]);
    bs_sbox(rot[16], rot[17], rot[18], rot[19],
            rot[20], rot[21], rot[22], rot[23]);
    bs_sbox(rot[24], rot[25], rot[26], rot[27],
            rot[28], rot[29], rot[30], rot[31]);

    // ── Step 3: XOR Rcon[rnd] into first byte of rot (rot[0..7]) ────────
    const uint8_t rcon = BS_KS_RCON[rnd];
    for (int b = 0; b < 8; b++) {
        if ((rcon >> b) & 1u) rot[b] ^= 0xFFFFFFFFu;
    }

    // ── Step 4: Expand each word in-place ────────────────────────────────
    // new_W0 (bytes  0-3):  W0 XOR rot
    for (int i = 0; i < 32; i++) rk[     i] ^= rot[i];
    // new_W1 (bytes  4-7):  W1 XOR new_W0
    for (int i = 0; i < 32; i++) rk[32 + i] ^= rk[     i];
    // new_W2 (bytes  8-11): W2 XOR new_W1
    for (int i = 0; i < 32; i++) rk[64 + i] ^= rk[32 + i];
    // new_W3 (bytes 12-15): W3 XOR new_W2
    for (int i = 0; i < 32; i++) rk[96 + i] ^= rk[64 + i];
}
