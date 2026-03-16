#pragma once
#include <stdint.h>

__device__ __forceinline__ void bs_mixcl(
    uint32_t &r0,  uint32_t &r1,  uint32_t &r2,  uint32_t &r3,
    uint32_t &r4,  uint32_t &r5,  uint32_t &r6,  uint32_t &r7,
    uint32_t &r8,  uint32_t &r9,  uint32_t &r10, uint32_t &r11,
    uint32_t &r12, uint32_t &r13, uint32_t &r14, uint32_t &r15,
    uint32_t &r16, uint32_t &r17, uint32_t &r18, uint32_t &r19,
    uint32_t &r20, uint32_t &r21, uint32_t &r22, uint32_t &r23,
    uint32_t &r24, uint32_t &r25, uint32_t &r26, uint32_t &r27,
    uint32_t &r28, uint32_t &r29, uint32_t &r30, uint32_t &r31)
{
    uint32_t o0=0,o1=0,o2=0,o3=0,o4=0,o5=0,o6=0,o7=0;
    uint32_t o8=0,o9=0,o10=0,o11=0,o12=0,o13=0,o14=0,o15=0;
    uint32_t o16=0,o17=0,o18=0,o19=0,o20=0,o21=0,o22=0,o23=0;
    uint32_t o24=0,o25=0,o26=0,o27=0,o28=0,o29=0,o30=0,o31=0;

    uint32_t a0,a1,a2,a3,a4,a5,a6,a7;
    uint32_t tmp;

    a0=r0 ^ r8;  a1=r1 ^ r9;  a2=r2 ^ r10; a3=r3 ^ r11;
    a4=r4 ^ r12; a5=r5 ^ r13; a6=r6 ^ r14; a7=r7 ^ r15;
    tmp=a7;
    a7=a6; a6=a5; a5=a4;
    a4=a3 ^ tmp; a3=a2 ^ tmp; a2=a1; a1=a0 ^ tmp; a0=tmp;

    o0=a0 ^ r8 ^ r16 ^ r24;   o1=a1 ^ r9 ^ r17 ^ r25;
    o2=a2 ^ r10 ^ r18 ^ r26;  o3=a3 ^ r11 ^ r19 ^ r27;
    o4=a4 ^ r12 ^ r20 ^ r28;  o5=a5 ^ r13 ^ r21 ^ r29;
    o6=a6 ^ r14 ^ r22 ^ r30;  o7=a7 ^ r15 ^ r23 ^ r31;

    a0=r8 ^ r16;  a1=r9 ^ r17;  a2=r10 ^ r18; a3=r11 ^ r19;
    a4=r12 ^ r20; a5=r13 ^ r21; a6=r14 ^ r22; a7=r15 ^ r23;
    tmp=a7;
    a7=a6; a6=a5; a5=a4;
    a4=a3 ^ tmp; a3=a2 ^ tmp; a2=a1; a1=a0 ^ tmp; a0=tmp;

    o8 =a0 ^ r0 ^ r16 ^ r24;  o9 =a1 ^ r1 ^ r17 ^ r25;
    o10=a2 ^ r2 ^ r18 ^ r26;  o11=a3 ^ r3 ^ r19 ^ r27;
    o12=a4 ^ r4 ^ r20 ^ r28;  o13=a5 ^ r5 ^ r21 ^ r29;
    o14=a6 ^ r6 ^ r22 ^ r30;  o15=a7 ^ r7 ^ r23 ^ r31;

    a0=r16 ^ r24; a1=r17 ^ r25; a2=r18 ^ r26; a3=r19 ^ r27;
    a4=r20 ^ r28; a5=r21 ^ r29; a6=r22 ^ r30; a7=r23 ^ r31;
    tmp=a7;
    a7=a6; a6=a5; a5=a4;
    a4=a3 ^ tmp; a3=a2 ^ tmp; a2=a1; a1=a0 ^ tmp; a0=tmp;

    o16=a0 ^ r8 ^ r0 ^ r24;   o17=a1 ^ r9 ^ r1 ^ r25;
    o18=a2 ^ r10 ^ r2 ^ r26;  o19=a3 ^ r11 ^ r3 ^ r27;
    o20=a4 ^ r12 ^ r4 ^ r28;  o21=a5 ^ r13 ^ r5 ^ r29;
    o22=a6 ^ r14 ^ r6 ^ r30;  o23=a7 ^ r15 ^ r7 ^ r31;

    a0=r0 ^ r24;  a1=r1 ^ r25;  a2=r2 ^ r26;  a3=r3 ^ r27;
    a4=r4 ^ r28;  a5=r5 ^ r29;  a6=r6 ^ r30;  a7=r7 ^ r31;
    tmp=a7;
    a7=a6; a6=a5; a5=a4;
    a4=a3 ^ tmp; a3=a2 ^ tmp; a2=a1; a1=a0 ^ tmp; a0=tmp;

    o24=a0 ^ r8 ^ r0 ^ r16;   o25=a1 ^ r9 ^ r1 ^ r17;
    o26=a2 ^ r10 ^ r2 ^ r18;  o27=a3 ^ r11 ^ r3 ^ r19;
    o28=a4 ^ r12 ^ r4 ^ r20;  o29=a5 ^ r13 ^ r5 ^ r21;
    o30=a6 ^ r14 ^ r6 ^ r22;  o31=a7 ^ r15 ^ r7 ^ r23;

    r0=o0;   r1=o1;   r2=o2;   r3=o3;   r4=o4;   r5=o5;   r6=o6;   r7=o7;
    r8=o8;   r9=o9;   r10=o10; r11=o11; r12=o12; r13=o13; r14=o14; r15=o15;
    r16=o16; r17=o17; r18=o18; r19=o19; r20=o20; r21=o21; r22=o22; r23=o23;
    r24=o24; r25=o25; r26=o26; r27=o27; r28=o28; r29=o29; r30=o30; r31=o31;
}

// Map 4 byte indices (each contributes 8 slices) to bs_mixcl parameters
template<int B0, int B1, int B2, int B3>
__device__ __forceinline__ void apply_mixcol(uint32_t (&r)[128]) {
    constexpr int a = B0*8, b = B1*8, c = B2*8, d = B3*8;
    bs_mixcl(
        r[a+0], r[a+1], r[a+2], r[a+3], r[a+4], r[a+5], r[a+6], r[a+7],
        r[b+0], r[b+1], r[b+2], r[b+3], r[b+4], r[b+5], r[b+6], r[b+7],
        r[c+0], r[c+1], r[c+2], r[c+3], r[c+4], r[c+5], r[c+6], r[c+7],
        r[d+0], r[d+1], r[d+2], r[d+3], r[d+4], r[d+5], r[d+6], r[d+7]
    );
}


// Gather SR column <S0,S1,S2,S3> into 32 locals, run bs_mixcl, scatter into
// contiguous dest bytes starting at DBASE (DBASE, DBASE+1, DBASE+2, DBASE+3).
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

