#pragma once
#include <stdint.h>
#include <stdio.h>

/**
 * @brief Helper function to debug a single column (4 bytes) of the bitsliced state.
 * 
 * Extracts the value for lane 0 and prints it.
 * 
 * @param label A label to identify the debug output.
 * @param r0 bit-slice 0 of the column.
 * @param r1 bit-slice 1 of the column.
 * @param r2 bit-slice 2 of the column.
 * @param r3 bit-slice 3 of the column.
 * @param r4 bit-slice 4 of the column.
 * @param r5 bit-slice 5 of the column.
 * @param r6 bit-slice 6 of the column.
 * @param r7 bit-slice 7 of the column.
 * @param r8 bit-slice 8 of the column.
 * @param r9 bit-slice 9 of the column.
 * @param r10 bit-slice 10 of the column.
 * @param r11 bit-slice 11 of the column.
 * @param r12 bit-slice 12 of the column.
 * @param r13 bit-slice 13 of the column.
 * @param r14 bit-slice 14 of the column.
 * @param r15 bit-slice 15 of the column.
 * @param r16 bit-slice 16 of the column.
 * @param r17 bit-slice 17 of the column.
 * @param r18 bit-slice 18 of the column.
 * @param r19 bit-slice 19 of the column.
 * @param r20 bit-slice 20 of the column.
 * @param r21 bit-slice 21 of the column.
 * @param r22 bit-slice 22 of the column.
 * @param r23 bit-slice 23 of the column.
 * @param r24 bit-slice 24 of the column.
 * @param r25 bit-slice 25 of the column.
 * @param r26 bit-slice 26 of the column.
 * @param r27 bit-slice 27 of the column.
 * @param r28 bit-slice 28 of the column.
 * @param r29 bit-slice 29 of the column.
 * @param r30 bit-slice 30 of the column.
 * @param r31 bit-slice 31 of the column.
 * @return void
 */
__device__ __forceinline__ void bs_debug_col(const char* label,
    uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3,
    uint32_t r4, uint32_t r5, uint32_t r6, uint32_t r7,
    uint32_t r8, uint32_t r9, uint32_t r10, uint32_t r11,
    uint32_t r12, uint32_t r13, uint32_t r14, uint32_t r15,
    uint32_t r16, uint32_t r17, uint32_t r18, uint32_t r19,
    uint32_t r20, uint32_t r21, uint32_t r22, uint32_t r23,
    uint32_t r24, uint32_t r25, uint32_t r26, uint32_t r27,
    uint32_t r28, uint32_t r29, uint32_t r30, uint32_t r31)
{
    // Lane 0만 출력
    uint8_t b[4] = {0,0,0,0};
    // 안전하게 수동으로 하나씩 복원
    { uint8_t v=0; if(r0&1)v|=1; if(r1&1)v|=2; if(r2&1)v|=4; if(r3&1)v|=8; if(r4&1)v|=16; if(r5&1)v|=32; if(r6&1)v|=64; if(r7&1)v|=128; b[0]=v; }
    { uint8_t v=0; if(r8&1)v|=1; if(r9&1)v|=2; if(r10&1)v|=4; if(r11&1)v|=8; if(r12&1)v|=16; if(r13&1)v|=32; if(r14&1)v|=64; if(r15&1)v|=128; b[1]=v; }
    { uint8_t v=0; if(r16&1)v|=1; if(r17&1)v|=2; if(r18&1)v|=4; if(r19&1)v|=8; if(r20&1)v|=16; if(r21&1)v|=32; if(r22&1)v|=64; if(r23&1)v|=128; b[2]=v; }
    { uint8_t v=0; if(r24&1)v|=1; if(r25&1)v|=2; if(r26&1)v|=4; if(r27&1)v|=8; if(r28&1)v|=16; if(r29&1)v|=32; if(r30&1)v|=64; if(r31&1)v|=128; b[3]=v; }

    printf("    [DEBUG_MC] %-10s: %02x %02x %02x %02x\n", label, b[0], b[1], b[2], b[3]);
}

/**
 * @brief Core bitsliced MixColumns implementation for a single column.
 * 
 * Performs the MixColumns transformation on 4 bytes (32 slices) in-place.
 * 
 * @param r0 bit-slice 0 of the column.
 * @param r1 bit-slice 1 of the column.
 * @param r2 bit-slice 2 of the column.
 * @param r3 bit-slice 3 of the column.
 * @param r4 bit-slice 4 of the column.
 * @param r5 bit-slice 5 of the column.
 * @param r6 bit-slice 6 of the column.
 * @param r7 bit-slice 7 of the column.
 * @param r8 bit-slice 8 of the column.
 * @param r9 bit-slice 9 of the column.
 * @param r10 bit-slice 10 of the column.
 * @param r11 bit-slice 11 of the column.
 * @param r12 bit-slice 12 of the column.
 * @param r13 bit-slice 13 of the column.
 * @param r14 bit-slice 14 of the column.
 * @param r15 bit-slice 15 of the column.
 * @param r16 bit-slice 16 of the column.
 * @param r17 bit-slice 17 of the column.
 * @param r18 bit-slice 18 of the column.
 * @param r19 bit-slice 19 of the column.
 * @param r20 bit-slice 20 of the column.
 * @param r21 bit-slice 21 of the column.
 * @param r22 bit-slice 22 of the column.
 * @param r23 bit-slice 23 of the column.
 * @param r24 bit-slice 24 of the column.
 * @param r25 bit-slice 25 of the column.
 * @param r26 bit-slice 26 of the column.
 * @param r27 bit-slice 27 of the column.
 * @param r28 bit-slice 28 of the column.
 * @param r29 bit-slice 29 of the column.
 * @param r30 bit-slice 30 of the column.
 * @param r31 bit-slice 31 of the column.
 * @param do_print [in] Whether to print debug information (default: false).
 * @return void
 */
__device__ __forceinline__ void bs_mixcl(
    uint32_t &r0,  uint32_t &r1,  uint32_t &r2,  uint32_t &r3,
    uint32_t &r4,  uint32_t &r5,  uint32_t &r6,  uint32_t &r7,
    uint32_t &r8,  uint32_t &r9,  uint32_t &r10, uint32_t &r11,
    uint32_t &r12, uint32_t &r13, uint32_t &r14, uint32_t &r15,
    uint32_t &r16, uint32_t &r17, uint32_t &r18, uint32_t &r19,
    uint32_t &r20, uint32_t &r21, uint32_t &r22, uint32_t &r23,
    uint32_t &r24, uint32_t &r25, uint32_t &r26, uint32_t &r27,
    uint32_t &r28, uint32_t &r29, uint32_t &r30, uint32_t &r31,
    bool do_print = false)
{
    if (do_print) bs_debug_col("MC_START", r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31);

    uint32_t o0=0,o1=0,o2=0,o3=0,o4=0,o5=0,o6=0,o7=0;
    uint32_t o8=0,o9=0,o10=0,o11=0,o12=0,o13=0,o14=0,o15=0;
    uint32_t o16=0,o17=0,o18=0,o19=0,o20=0,o21=0,o22=0,o23=0;
    uint32_t o24=0,o25=0,o26=0,o27=0,o28=0,o29=0,o30=0,o31=0;

    uint32_t a0,a1,a2,a3,a4,a5,a6,a7;
    uint32_t tmp;

    // Row 0 update
    a0=r0 ^ r8;  a1=r1 ^ r9;  a2=r2 ^ r10; a3=r3 ^ r11;
    a4=r4 ^ r12; a5=r5 ^ r13; a6=r6 ^ r14; a7=r7 ^ r15;
    tmp=a7;
    a7=a6; a6=a5; a5=a4;
    a4=a3 ^ tmp; a3=a2 ^ tmp; a2=a1; a1=a0 ^ tmp; a0=tmp;

    o0=a0 ^ r8 ^ r16 ^ r24;   o1=a1 ^ r9 ^ r17 ^ r25;
    o2=a2 ^ r10 ^ r18 ^ r26;  o3=a3 ^ r11 ^ r19 ^ r27;
    o4=a4 ^ r12 ^ r20 ^ r28;  o5=a5 ^ r13 ^ r21 ^ r29;
    o6=a6 ^ r14 ^ r22 ^ r30;  o7=a7 ^ r15 ^ r23 ^ r31;
    if (do_print) bs_debug_col("AFTER_R0", o0,o1,o2,o3,o4,o5,o6,o7, r8,r9,r10,r11,r12,r13,r14,r15, r16,r17,r18,r19,r20,r21,r22,r23, r24,r25,r26,r27,r28,r29,r30,r31);

    // Row 1 update
    a0=r8 ^ r16;  a1=r9 ^ r17;  a2=r10 ^ r18; a3=r11 ^ r19;
    a4=r12 ^ r20; a5=r13 ^ r21; a6=r14 ^ r22; a7=r15 ^ r23;
    tmp=a7;
    a7=a6; a6=a5; a5=a4;
    a4=a3 ^ tmp; a3=a2 ^ tmp; a2=a1; a1=a0 ^ tmp; a0=tmp;

    o8 =a0 ^ r0 ^ r16 ^ r24;  o9 =a1 ^ r1 ^ r17 ^ r25;
    o10=a2 ^ r2 ^ r18 ^ r26;  o11=a3 ^ r3 ^ r19 ^ r27;
    o12=a4 ^ r4 ^ r20 ^ r28;  o13=a5 ^ r5 ^ r21 ^ r29;
    o14=a6 ^ r6 ^ r22 ^ r30;  o15=a7 ^ r7 ^ r23 ^ r31;
    if (do_print) bs_debug_col("AFTER_R1", o0,o1,o2,o3,o4,o5,o6,o7, o8,o9,o10,o11,o12,o13,o14,o15, r16,r17,r18,r19,r20,r21,r22,r23, r24,r25,r26,r27,r28,r29,r30,r31);

    // Row 2 update
    a0=r16 ^ r24; a1=r17 ^ r25; a2=r18 ^ r26; a3=r19 ^ r27;
    a4=r20 ^ r28; a5=r21 ^ r29; a6=r22 ^ r30; a7=r23 ^ r31;
    tmp=a7;
    a7=a6; a6=a5; a5=a4;
    a4=a3 ^ tmp; a3=a2 ^ tmp; a2=a1; a1=a0 ^ tmp; a0=tmp;

    o16=a0 ^ r8 ^ r0 ^ r24;   o17=a1 ^ r9 ^ r1 ^ r25;
    o18=a2 ^ r10 ^ r2 ^ r26;  o19=a3 ^ r11 ^ r3 ^ r27;
    o20=a4 ^ r12 ^ r4 ^ r28;  o21=a5 ^ r13 ^ r5 ^ r29;
    o22=a6 ^ r14 ^ r6 ^ r30;  o23=a7 ^ r15 ^ r7 ^ r31;
    if (do_print) bs_debug_col("AFTER_R2", o0,o1,o2,o3,o4,o5,o6,o7, o8,o9,o10,o11,o12,o13,o14,o15, o16,o17,o18,o19,o20,o21,o22,o23, r24,r25,r26,r27,r28,r29,r30,r31);

    // Row 3 update
    a0=r0 ^ r24;  a1=r1 ^ r25;  a2=r2 ^ r26;  a3=r3 ^ r27;
    a4=r4 ^ r28;  a5=r5 ^ r29;  a6=r6 ^ r30;  a7=r7 ^ r31;
    tmp=a7;
    a7=a6; a6=a5; a5=a4;
    a4=a3 ^ tmp; a3=a2 ^ tmp; a2=a1; a1=a0 ^ tmp; a0=tmp;

    o24=a0 ^ r8 ^ r0 ^ r16;   o25=a1 ^ r9 ^ r1 ^ r17;
    o26=a2 ^ r10 ^ r2 ^ r18;  o27=a3 ^ r11 ^ r3 ^ r19;
    o28=a4 ^ r12 ^ r4 ^ r20;  o29=a5 ^ r13 ^ r5 ^ r21;
    o30=a6 ^ r14 ^ r6 ^ r22;  o31=a7 ^ r15 ^ r7 ^ r23;
    if (do_print) bs_debug_col("AFTER_R3", o0,o1,o2,o3,o4,o5,o6,o7, o8,o9,o10,o11,o12,o13,o14,o15, o16,o17,o18,o19,o20,o21,o22,o23, o24,o25,o26,o27,o28,r29,r30,r31);

    r0=o0;   r1=o1;   r2=o2;   r3=o3;   r4=o4;   r5=o5;   r6=o6;   r7=o7;
    r8=o8;   r9=o9;   r10=o10; r11=o11; r12=o12; r13=o13; r14=o14; r15=o15;
    r16=o16; r17=o17; r18=o18; r19=o19; r20=o20; r21=o21; r22=o22; r23=o23;
    r24=o24; r25=o25; r26=o26; r27=o27; r28=o28; r29=o29; r30=o30; r31=o31;
}

/**
 * @brief Maps 4 byte indices to bs_mixcl parameters and applies MixColumns.
 * 
 * @tparam B0 The index of the first byte in the bitsliced state (0-15).
 * @tparam B1 The index of the second byte in the bitsliced state (0-15).
 * @tparam B2 The index of the third byte in the bitsliced state (0-15).
 * @tparam B3 The index of the fourth byte in the bitsliced state (0-15).
 * @param r [in,out] The 128-slice bitsliced state.
 * @param do_print [in] Whether to print debug information (default: false).
 * @return void
 */
template<int B0, int B1, int B2, int B3>
__device__ __forceinline__ void apply_mixcol(uint32_t (&r)[128], bool do_print = false) {
    constexpr int a = B0*8, b = B1*8, c = B2*8, d = B3*8;
    bs_mixcl(
        r[a+0], r[a+1], r[a+2], r[a+3], r[a+4], r[a+5], r[a+6], r[a+7],
        r[b+0], r[b+1], r[b+2], r[b+3], r[b+4], r[b+5], r[b+6], r[b+7],
        r[c+0], r[c+1], r[c+2], r[c+3], r[c+4], r[c+5], r[c+6], r[c+7],
        r[d+0], r[d+1], r[d+2], r[d+3], r[d+4], r[d+5], r[d+6], r[d+7],
        do_print
    );
}


/**
 * @brief Fused ShiftRows and MixColumns optimization.
 * 
 * Gathers bitsliced slices from specified indices (ShiftRows pattern),
 * applies MixColumns, and scatters back into contiguous destination bytes.
 * 
 * @tparam S0 First source byte index.
 * @tparam S1 Second source byte index.
 * @tparam S2 Third source byte index.
 * @tparam S3 Fourth source byte index.
 * @tparam DBASE Destination base byte index (DBASE, DBASE+1, DBASE+2, DBASE+3).
 * @param r [in,out] The 128-slice bitsliced state.
 * @return void
 */
template<int S0, int S1, int S2, int S3, int DBASE>
__device__ __forceinline__ void mc_sr_fused(uint32_t (&r)[128]) {
    // 32 locals (one column = 4 bytes × 8 slices)
    uint32_t x0=r[S0*8+0],  x1=r[S0*8+1],  x2=r[S0*8+2],  x3=r[S0*8+3],
             x4=r[S0*8+4],  x5=r[S0*8+5],  x6=r[S0*8+6],  x7=r[S0*8+7];
    uint32_t x8=r[S1*8+0],  x9=r[S1*8+1],  x10=r[S1*8+2], x11=r[S1*8+3],
             x12=r[S1*8+4], x13=r[S1*8+5], x14=r[S1*8+6], x15=r[S1*8+7];
    uint32_t x16=r[S2*8+0], x17=r[S2*8+1], x18=r[S2*8+2], x19=r[S2*8+3],
             x20=r[S2*8+4], x21=r[S2*8+5], x22=r[S2*8+6], x23=r[S2*8+7];
    uint32_t x24=r[S3*8+0], x25=r[S3*8+1], x26=r[S3*8+2], x27=r[S3*8+3],
             x28=r[S3*8+4], x29=r[S3*8+5], x30=r[S3*8+6], x31=r[S3*8+7];

    // Compute MixColumns in-place on the locals
    bs_mixcl(
        x0,x1,x2,x3,x4,x5,x6,x7,
        x8,x9,x10,x11,x12,x13,x14,x15,
        x16,x17,x18,x19,x20,x21,x22,x23,
        x24,x25,x26,x27,x28,x29,x30,x31
    );

    // Scatter into *contiguous* bytes DBASE..DBASE+3 (SR layout)
    r[(DBASE+0)*8+0]=x0;  r[(DBASE+0)*8+1]=x1;  r[(DBASE+0)*8+2]=x2;  r[(DBASE+0)*8+3]=x3;
    r[(DBASE+0)*8+4]=x4;  r[(DBASE+0)*8+5]=x5;  r[(DBASE+0)*8+6]=x6;  r[(DBASE+0)*8+7]=x7;

    r[(DBASE+1)*8+0]=x8;  r[(DBASE+1)*8+1]=x9;  r[(DBASE+1)*8+2]=x10; r[(DBASE+1)*8+3]=x11;
    r[(DBASE+1)*8+4]=x12; r[(DBASE+1)*8+5]=x13; r[(DBASE+1)*8+6]=x14; r[(DBASE+1)*8+7]=x15;

    r[(DBASE+2)*8+0]=x16; r[(DBASE+2)*8+1]=x17; r[(DBASE+2)*8+2]=x18; r[(DBASE+2)*8+3]=x19;
    r[(DBASE+2)*8+4]=x20; r[(DBASE+2)*8+5]=x21; r[(DBASE+2)*8+6]=x22; r[(DBASE+2)*8+7]=x23;

    r[(DBASE+3)*8+0]=x24; r[(DBASE+3)*8+1]=x25; r[(DBASE+3)*8+2]=x26; r[(DBASE+3)*8+3]=x27;
    r[(DBASE+3)*8+4]=x28; r[(DBASE+3)*8+5]=x29; r[(DBASE+3)*8+6]=x30; r[(DBASE+3)*8+7]=x31;
}
