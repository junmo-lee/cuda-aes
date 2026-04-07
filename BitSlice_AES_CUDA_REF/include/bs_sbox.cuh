#pragma once
#include <stdint.h>

/**
 * @brief Bitsliced AES S-box implementation.
 * 
 * Performs the S-box transformation in-place on eight 32-bit lane packs (32 lanes).
 * This is a highly optimized Boolean circuit implementation.
 * 
 * @param r0..r7 [in,out] The 8 bit-slices representing the byte to be transformed.
 */
__device__ __forceinline__ void bs_sbox(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
                                        uint32_t &r4, uint32_t &r5, uint32_t &r6, uint32_t &r7) {
    uint32_t y0 = (r0 ^ r1 ^ r2 ^ r3 ^ r6);
    uint32_t y1 = (r0 ^ r5 ^ r6);
    uint32_t y2 = (r0);
    uint32_t y3 = (r0 ^ r1 ^ r3 ^ r4 ^ r7);
    uint32_t y4 = (r0 ^ r5 ^ r6 ^ r7);
    uint32_t y5 = (r0 ^ r1 ^ r5 ^ r6);
    uint32_t y6 = (r0 ^ r4 ^ r5 ^ r6);
    uint32_t y7 = (r0 ^ r1 ^ r2 ^ r5 ^ r6 ^ r7);

    uint32_t v0 = y0 ^ y4 ^ y6 ^ y2;
    uint32_t v1 = y1 ^ y5 ^ y7 ^ y3;
    uint32_t m0 = y1 ^ y5;
    uint32_t m1 = y0 ^ y4;
    uint32_t c0 = m1;
    uint32_t c1 = m0 ^ m1;
    uint32_t c2 = v1;
    uint32_t c3 = v0;
    uint32_t e  = ( (y4 ^ y5 ^ y6 ^ y7) & (y0 ^ y1 ^ y2 ^ y3) );
    uint32_t e0 = ((y6 ^ y4) & (y0 ^ y2)) ^ e;
    uint32_t e1 = ((y5 ^ y7) & (y1 ^ y3)) ^ e;
    uint32_t d1 = e1;
    e1 = e0;
    e0 = d1 ^ e0;

    uint32_t p  = (y7 ^ y6) & (y2 ^ y3);
    uint32_t d2 = (y6 & y2) ^ p ^ e0;
    uint32_t d3 = (y7 & y3) ^ p ^ e1;
    uint32_t q  = (y5 ^ y4) & (y1 ^ y0);
    uint32_t d0 = (y4 & y0) ^ q ^ e0;
    d1 = (y5 & y1) ^ q ^ e1;
    v0 = c2 ^ d2;
    v1 = c3 ^ d3;
    m0 = c0 ^ d0;
    m1 = c1 ^ d1;
    d0 = v1 ^ m1;
    d1 = v0 ^ m0;
    c1 = d0;
    c0 = d1 ^ d0;

    uint32_t d_ = (v1 ^ v0) & (m1 ^ m0);
    d0 = (v0 & m0) ^ d_;
    d1 = (v1 & m1) ^ d_;
    e0 = c1 ^ d1;
    e1 = c0 ^ d0;
    p  = (e1 ^ e0) & (m1 ^ m0);
    uint32_t p0 = (e0 & m0) ^ p;
    uint32_t p1 = (e1 & m1) ^ p;
    q  = (e1 ^ e0) & (v1 ^ v0);
    uint32_t q0 = (e0 & v0) ^ q;
    uint32_t q1 = (e1 & v1) ^ q;
    uint32_t e3 = p1;
    uint32_t e2 = p0;
    e1 = q1;
    e0 = q0;
    v0 = e2;
    v1 = e3;
    m0 = e0;
    m1 = e1;

    uint32_t c  = (v1 ^ m1 ^ v0 ^ m0) & (y0 ^ y1 ^ y2 ^ y3);
    c0 = ((v0 ^ m0) & (y2 ^ y0)) ^ c;
    c1 = ((v1 ^ m1) & (y3 ^ y1)) ^ c;
    uint32_t temp = c1;
    c1 = c0;
    c0 = temp ^ c0;
    p  = (v1 ^ v0) & (y3 ^ y2);
    uint32_t p2 = (v0 & y2) ^ p ^ c0;
    uint32_t p3 = (v1 & y3) ^ p ^ c1;
    q  = (m1 ^ m0) & (y1 ^ y0);
    p0 = (m0 & y0) ^ q ^ c0;
    p1 = (m1 & y1) ^ q ^ c1;
    uint32_t d1x = e1;
    uint32_t d0x = e0;
    c  = (y4 ^ y5 ^ y6 ^ y7) & (e0 ^ e1 ^ e2 ^ e3);
    c0 = ((y6 ^ y4) & (e2 ^ e0)) ^ c;
    c1 = ((y7 ^ y5) & (e3 ^ e1)) ^ c;
    temp = c1;
    c1 = c0;
    c0 = temp ^ c0;
    e  = (y7 ^ y6) & (e2 ^ e3);
    e0 = (y6 & e2) ^ e ^ c0;
    e1 = (y7 & e3) ^ e ^ c1;
    q  = (y5 ^ y4) & (d0x ^ d1x);
    uint32_t q0x = (y4 & d0x) ^ q ^ c0;
    uint32_t q1x = (y5 & d1x) ^ q ^ c1;

    uint32_t nr0 = (q1x ^ p0 ^ p2 ^ 0xFFFFFFFFu);
    uint32_t nr1 = (q1x ^ p0 ^ p1 ^ 0xFFFFFFFFu);
    uint32_t nr2 = (q0x ^ e0 ^ e1 ^ p1 ^ p2);
    uint32_t nr3 = (e1 ^ p0 ^ p1 ^ p2 ^ p3);
    uint32_t nr4 = (e1 ^ p1 ^ p3);
    uint32_t nr5 = (q0x ^ p2 ^ 0xFFFFFFFFu);
    uint32_t nr6 = (e1 ^ p3 ^ 0xFFFFFFFFu);
    uint32_t nr7 = (e1 ^ p1);

    r0 = nr0; r1 = nr1; r2 = nr2; r3 = nr3;
    r4 = nr4; r5 = nr5; r6 = nr6; r7 = nr7;
}
