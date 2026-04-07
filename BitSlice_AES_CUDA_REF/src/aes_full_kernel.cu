#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio> 
#include <vector>

// --- your project headers ---
#include "bs_sbox.cuh"
#include "bs_mixcol.cuh"
#include "bs_helpers.cuh"
#include "ct_key.hpp"
#include "aes_keys.hpp"

#ifdef USE_GENERATED_RK
  #include "generated_rk.hpp"   // provides MyKey and rkbit_const<ROUND,SLICE>::value
#else
  /**
   * @brief Simple tag type for compile-time key bytes (fallback).
   * 
   * @tparam Ks The key bytes.
   */
  template<unsigned char... Ks> struct StaticKey {
      static constexpr unsigned char data[sizeof...(Ks)] = { Ks... };
  };
  // AES-128 test key: 2b7e...4f3c
  using MyKey = StaticKey<
      0x2B,0x7E,0x15,0x16, 0x28,0xAE,0xD2,0xA6,
      0xAB,0xF7,0x15,0x88, 0x09,0xCF,0x4F,0x3C
  >;
#endif

/**
 * @brief Round-key bit at compile time.
 * 
 * @tparam ROUND The round number (0-10).
 * @tparam SLICE The bit slice index (0-127).
 */
template<int ROUND, int SLICE>
struct RKBitConst {
#ifdef USE_GENERATED_RK
    static constexpr bool value = rkbit_const<ROUND, SLICE>::value;
#else
    // Fallback: compute key schedule at compile time
    static constexpr bool value =
        Aes128KeySchedule<
            MyKey::data[0],  MyKey::data[1],  MyKey::data[2],  MyKey::data[3],
            MyKey::data[4],  MyKey::data[5],  MyKey::data[6],  MyKey::data[7],
            MyKey::data[8],  MyKey::data[9],  MyKey::data[10], MyKey::data[11],
            MyKey::data[12], MyKey::data[13], MyKey::data[14], MyKey::data[15]
        >::template rk_bit<ROUND, SLICE>();
#endif
};


/**
 * @brief Performs the AES ShiftRows transformation in-place on a bitsliced state.
 * 
 * @param r [in,out] The 128-slice bitsliced state.
 */
__device__ __forceinline__ void shiftrows_inplace(uint32_t (&r)[128]) {
    uint32_t t;
    // row1 rotate left by 1: 1,5,9,13
    #pragma unroll
    for (int k=0;k<8;++k){ t=r[1*8+k]; r[1*8+k]=r[5*8+k]; r[5*8+k]=r[9*8+k]; r[9*8+k]=r[13*8+k]; r[13*8+k]=t; }
    // row2 rotate left by 2: 2<->10, 6<->14
    #pragma unroll
    for (int k=0;k<8;++k){ t=r[2*8+k]; r[2*8+k]=r[10*8+k]; r[10*8+k]=t; t=r[6*8+k]; r[6*8+k]=r[14*8+k]; r[14*8+k]=t; }
    // row3 rotate left by 3 (right by 1): 3,7,11,15
    #pragma unroll
    for (int k=0;k<8;++k){ t=r[15*8+k]; r[15*8+k]=r[11*8+k]; r[11*8+k]=r[7*8+k]; r[7*8+k]=r[3*8+k]; r[3*8+k]=t; }
}


/**
 * @brief Stores the bitsliced state to memory while applying the ShiftRows transformation.
 * 
 * @param r [in] The 128-slice bitsliced state.
 * @param out [out] Pointer to the output memory.
 */
__device__ __forceinline__
void store128_shiftrows(const uint32_t (&r)[128], uint32_t* out) {
    // SR byte map (SR index -> pre-SR source idx)
    constexpr int map[16] = {0,5,10,15, 4,9,14,3, 8,13,2,7, 12,1,6,11};
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int src = map[i] * 8;
        const int dst = i * 8;
        out[dst+0]=r[src+0]; out[dst+1]=r[src+1]; out[dst+2]=r[src+2]; out[dst+3]=r[src+3];
        out[dst+4]=r[src+4]; out[dst+5]=r[src+5]; out[dst+6]=r[src+6]; out[dst+7]=r[src+7];
    }
}


/**
 * @brief XORs a single bit-slice with a compile-time round key bit.
 * 
 * @tparam ROUND The round number.
 * @tparam SLICE The bit slice index.
 * @param x [in,out] The 32-lane bit-slice.
 */
template<int ROUND, int SLICE>
__device__ __forceinline__ void addkey_slice(uint32_t &x) {
    if constexpr (RKBitConst<ROUND, SLICE>::value) {
        // XOR with key bit == 1 -> flip the slice
        x = ~x;
    }
}

/**
 * @brief XORs the entire 128-slice state with a compile-time round key.
 * 
 * @tparam ROUND The round number.
 * @tparam I The current slice index being processed.
 * @param r [in,out] The 128-slice bitsliced state.
 */
template<int ROUND, int I>
__device__ __forceinline__ void addkey_128(uint32_t (&r)[128]) {
    addkey_slice<ROUND, I>(r[I]);
    if constexpr (I + 1 < 128) addkey_128<ROUND, I+1>(r);
}

/**
 * @brief Recursively applies the bitsliced S-box to all 16 bytes.
 * 
 * @tparam BYTE The current byte index (starts at 0).
 * @param r [in,out] The 128-slice bitsliced state.
 */
template<int BYTE>
__device__ __forceinline__ void subbytes_16(uint32_t (&r)[128]) {
    constexpr int base = BYTE * 8;
    bs_sbox(r[base+0], r[base+1], r[base+2], r[base+3],
            r[base+4], r[base+5], r[base+6], r[base+7]);
    if constexpr (BYTE + 1 < 16) subbytes_16<BYTE+1>(r);
}

/**
 * @brief Standard AES round implementation (Round 1-9) using implicit ShiftRows.
 * 
 * @tparam KS Key schedule type.
 * @param r [in,out] The 128-slice bitsliced state.
 * @param round_idx The current round number.
 */
template<typename KS>
__device__ __forceinline__ void round_main(uint32_t (&r)[128], int round_idx) {
    subbytes_16<0>(r);
    // MixColumns over SR-mapped columns (no explicit SR buffer)
    apply_mixcol<0,5,10,15>(r);
    apply_mixcol<4,9,14,3>(r);
    apply_mixcol<8,13,2,7>(r);
    apply_mixcol<12,1,6,11>(r);

    switch (round_idx) {
        case 1: addkey_128<1,0>(r); break;
        case 2: addkey_128<2,0>(r); break;
        case 3: addkey_128<3,0>(r); break;
        case 4: addkey_128<4,0>(r); break;
        case 5: addkey_128<5,0>(r); break;
        case 6: addkey_128<6,0>(r); break;
        case 7: addkey_128<7,0>(r); break;
        case 8: addkey_128<8,0>(r); break;
        case 9: addkey_128<9,0>(r); break;
        default: break;
    }
}

/**
 * @brief Final AES round implementation (Round 10).
 * 
 * @tparam KS Key schedule type.
 * @param r [in,out] The 128-slice bitsliced state.
 */
template<typename KS>
__device__ __forceinline__ void round_final(uint32_t (&r)[128]) {
    subbytes_16<0>(r);
    // Materialize ShiftRows view into a temp, then ARK10
    uint32_t t[128];
    constexpr int map[16] = {0,5,10,15, 4,9,14,3, 8,13,2,7, 12,1,6,11};
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int src = map[i] * 8;
        const int dst = i * 8;
        t[dst+0]=r[src+0]; t[dst+1]=r[src+1]; t[dst+2]=r[src+2]; t[dst+3]=r[src+3];
        t[dst+4]=r[src+4]; t[dst+5]=r[src+5]; t[dst+6]=r[src+6]; t[dst+7]=r[src+7];
    }
    addkey_128<10,0>(t);
    #pragma unroll
    for (int i = 0; i < 128; ++i) r[i] = t[i];
}

/**
 * @brief AES round implementation using in-place ShiftRows.
 * 
 * @tparam KS Key schedule type.
 * @param r [in,out] The 128-slice bitsliced state.
 * @param round_idx The current round number.
 */
template<typename KS>
__device__ __forceinline__ void round_main_sr_inplace(uint32_t (&r)[128], int round_idx) {
    subbytes_16<0>(r);        // S-box on all 16 bytes (bitsliced)
    shiftrows_inplace(r);     // <- in-place SR (low register pressure)

    // Now columns are contiguous: {0..3}, {4..7}, {8..11}, {12..15}
    apply_mixcol<0,1,2,3>(r);
    apply_mixcol<4,5,6,7>(r);
    apply_mixcol<8,9,10,11>(r);
    apply_mixcol<12,13,14,15>(r);

    // AddRoundKey (keys are for SR layout; state is already SR-after-SB)
    switch (round_idx) {
        case 1: addkey_128<1,0>(r);  break; case 2: addkey_128<2,0>(r);  break;
        case 3: addkey_128<3,0>(r);  break; case 4: addkey_128<4,0>(r);  break;
        case 5: addkey_128<5,0>(r);  break; case 6: addkey_128<6,0>(r);  break;
        case 7: addkey_128<7,0>(r);  break; case 8: addkey_128<8,0>(r);  break;
        case 9: addkey_128<9,0>(r);  break;
    }
}

/**
 * @brief AES round implementation using fused ShiftRows and MixColumns.
 * 
 * @tparam KS Key schedule type.
 * @param r [in,out] The 128-slice bitsliced state.
 * @param round_idx The current round number.
 */
template<typename KS>
__device__ __forceinline__ void round_main_fused(uint32_t (&r)[128], int round_idx) {
    subbytes_16<0>(r);
    // Fused SR→MC, keep state in SR layout after this:
    mc_sr_fused<0,5,10,15,   0>(r);
    mc_sr_fused<4,9,14,3,    4>(r);
    mc_sr_fused<8,13,2,7,    8>(r);
    mc_sr_fused<12,1,6,11,  12>(r);

    // ARK_rnd (keys expect SR layout)
    switch (round_idx) {
        case 1: addkey_128<1,0>(r);  break;  case 2: addkey_128<2,0>(r);  break;
        case 3: addkey_128<3,0>(r);  break;  case 4: addkey_128<4,0>(r);  break;
        case 5: addkey_128<5,0>(r);  break;  case 6: addkey_128<6,0>(r);  break;
        case 7: addkey_128<7,0>(r);  break;  case 8: addkey_128<8,0>(r);  break;
        case 9: addkey_128<9,0>(r);  break;
    }
}

/**
 * @brief Final AES round implementation using in-place ShiftRows.
 * 
 * @tparam KS Key schedule type.
 * @param r [in,out] The 128-slice bitsliced state.
 * @param out [out] Pointer to the output memory.
 */
template<typename KS>
__device__ __forceinline__ void round_final_sr_inplace(uint32_t (&r)[128], uint32_t* out) {
    subbytes_16<0>(r);
    shiftrows_inplace(r);
    addkey_128<10,0>(r);      // final round key applied to SR state
    store128<0>(r, out);      // state already in SR byte order; store directly
}


// -------- Debug dump plumbing --------
#ifndef DBG_DUMPS_PER_GROUP
#define DBG_DUMPS_PER_GROUP 11   // ARK@0..10
#endif

/**
 * @brief Dumps the current bitsliced state to a debug buffer.
 * 
 * @param dbg Pointer to the debug buffer.
 * @param group_idx The index of the lane group.
 * @param stage The current stage (0..39).
 * @param r The current bitsliced state.
 */
__device__ __forceinline__
void dump_stage(uint32_t* dbg, unsigned long long group_idx, int stage, const uint32_t r[128]) {
    if (!dbg) return;
    uint32_t* base = dbg + (group_idx * DBG_DUMPS_PER_GROUP + stage) * 128ull;
    #pragma unroll
    for (int i=0; i<128; ++i) base[i] = r[i];
}

/**
 * @brief Conditionally dumps the current bitsliced state if it matches the target stage.
 * 
 * @param dbg Pointer to the debug buffer.
 * @param gid The global ID of the lane group.
 * @param stage The current stage.
 * @param target The target stage to dump.
 * @param r The current bitsliced state.
 */
__device__ __forceinline__
void dump_if(uint32_t* dbg, unsigned long long gid, int stage, int target, const uint32_t r[128]) {
    if (!dbg || stage != target) return;
    uint32_t* base = dbg + gid * 128ull;
    #pragma unroll
    for (int i=0;i<128;++i) base[i] = r[i];
}

/**
 * @brief Creates a ShiftRows view of the bitsliced state into a temporary buffer.
 * 
 * @param r [in] The source bitsliced state.
 * @param t [out] The destination bitsliced state (ShiftRows order).
 */
__device__ __forceinline__
void make_shiftrows_view(const uint32_t r[128], uint32_t t[128]) {
    // bytes map after SR: {0,5,10,15, 4,9,14,3, 8,13,2,7, 12,1,6,11}
    constexpr int map[16] = {0,5,10,15, 4,9,14,3, 8,13,2,7, 12,1,6,11};
    #pragma unroll
    for (int i=0;i<16;++i) {
        int src = map[i]*8, dst=i*8;
        t[dst+0]=r[src+0]; t[dst+1]=r[src+1]; t[dst+2]=r[src+2]; t[dst+3]=r[src+3];
        t[dst+4]=r[src+4]; t[dst+5]=r[src+5]; t[dst+6]=r[src+6]; t[dst+7]=r[src+7];
    }
}

/**
 * @brief CUDA kernel for bitsliced AES-128 encryption with debug dumping.
 * 
 * @tparam KeyType The type containing the AES key bytes.
 * @param in_slices Input bitsliced state.
 * @param out_slices Output bitsliced state (ciphertext).
 * @param groups Number of 128-slice groups.
 * @param dbg_slices Debug output buffer.
 * @param target_stage The stage to dump (0-39).
 */
template<typename KeyType>
__global__ void aes128_bs32_full_debug(const uint32_t* __restrict__ in_slices,
                                 uint32_t* __restrict__ out_slices,
                                 unsigned long long groups,
                                 uint32_t* __restrict__ dbg_slices, // points to groups*128 u32
                                 int target_stage)                  // which step to dump
{
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= groups) return;
    using KS = typename MakeSchedule<KeyType>::type;

    const uint32_t* in  = in_slices  + tid * 128ull;
    uint32_t*       out = out_slices + tid * 128ull;
    uint32_t r[128]; load128<0>(in, r);

    int st = 0;

    // ---- Round 0: ARK0
    addkey_128<0,0>(r);
    dump_if(dbg_slices, tid, st++, target_stage, r); // ARK0

    // ---- Rounds 1..9 (SB -> SR -> MC -> ARK)
    for (int rnd=1; rnd<=9; ++rnd) {
        subbytes_16<0>(r);
        dump_if(dbg_slices, tid, st++, target_stage, r); // SB_r

        uint32_t sr[128]; make_shiftrows_view(r, sr);
        dump_if(dbg_slices, tid, st++, target_stage, sr); // SR_r view

        // mixcolumns uses SR ordering of bytes; either operate on 'sr' into r, or use your reindexed apply_mixcol
        // here we reuse your "implicit SR via indices" path by copying sr->r then MC in place:
        #pragma unroll
        for (int i=0;i<128;++i) r[i] = sr[i];
        apply_mixcol<0,1,2,3>(r);
        apply_mixcol<4,5,6,7>(r);
        apply_mixcol<8,9,10,11>(r);
        apply_mixcol<12,13,14,15>(r);
        dump_if(dbg_slices, tid, st++, target_stage, r); // MC_r

        // ARK_r
        switch (rnd) {
            case 1: addkey_128<1,0>(r); break; 
            case 2: addkey_128<2,0>(r); break;
            case 3: addkey_128<3,0>(r); break; 
            case 4: addkey_128<4,0>(r); break;
            case 5: addkey_128<5,0>(r); break; 
            case 6: addkey_128<6,0>(r); break;
            case 7: addkey_128<7,0>(r); break; 
            case 8: addkey_128<8,0>(r); break;
            case 9: addkey_128<9,0>(r); break;
        }
        dump_if(dbg_slices, tid, st++, target_stage, r); // ARK_r
    }

        // ---- Round 10 (final): SB -> SR -> ARK10 
        subbytes_16<0>(r); 
        dump_if(dbg_slices, tid, st++, target_stage, r); 
        // SB_10
        uint32_t sr10[128]; 
        make_shiftrows_view(r, sr10); 
        dump_if(dbg_slices, tid, st++, target_stage, sr10); 
        // SR_10 // ARK10 on SR10: 
        addkey_128<10,0>(sr10); 
        dump_if(dbg_slices, tid, st++, target_stage, sr10); 
        // ARK_10 (== ciphertext) // store ciphertext:
        store128<0>(sr10, out);
}

/**
 * @brief Standard CUDA kernel for bitsliced AES-128 encryption.
 * 
 * @tparam KeyType The type containing the AES key bytes.
 * @param in_slices Input bitsliced state.
 * @param out_slices Output bitsliced state (ciphertext).
 * @param groups Number of 128-slice groups.
 */
template<typename KeyType>
__global__ void aes128_bs32_full(const uint32_t* __restrict__ in_slices,
                                 uint32_t* __restrict__ out_slices,
                                 unsigned long long groups)
{
    const unsigned long long tid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= groups) return;

    using KS = typename MakeSchedule<KeyType>::type;

    const uint32_t* in  = in_slices  + tid * 128ull;
    uint32_t*       out = out_slices + tid * 128ull;

    uint32_t r[128];
    load128<0>(in, r);

    // Round 0 (initial)
    addkey_128<0,0>(r);

    // Rounds 1..9
    #pragma unroll
    for (int rnd = 1; rnd <= 9; ++rnd) {
        round_main_sr_inplace<KS>(r, rnd);
    }

    // Round 10 (no MixColumns)
    round_final_sr_inplace<KS>(r, out);
}


// ---------------- Host launcher ----------------
#define CUDA_OK(call) do {                                      \
    cudaError_t _e = (call);                                    \
    if (_e != cudaSuccess) {                                    \
        std::fprintf(stderr, "%s:%d CUDA error: %s\n",          \
                      __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return -1;                                              \
    }                                                           \
} while(0)

/**
 * @brief Host function to run the bitsliced AES-128 encryption.
 * 
 * @param h_in_u32 Host pointer to the input bitsliced state.
 * @param h_out_u32 Host pointer to store the output bitsliced state.
 * @param groups Number of 128-slice groups.
 * @param grid CUDA grid dimensions.
 * @param block CUDA block dimensions.
 * @return 0 on success, -1 on CUDA error.
 */
extern "C" int run_aes_bs_full(const uint32_t* h_in_u32,
                               uint32_t* h_out_u32,
                               unsigned long long groups,
                               dim3 grid, dim3 block)
{
    if (groups == 0) return 0;

    // Each group = 128 u32 slices
    const size_t N      = (size_t)groups * 128ull;
    const size_t nbytes = N * sizeof(uint32_t);

    // Threads launched
    const unsigned long long threads =
        (unsigned long long)grid.x * block.x *
        (unsigned long long)grid.y * block.y *
        (unsigned long long)grid.z * block.z;

    if (threads < groups) {
        std::fprintf(stderr, "WARNING: threads (%llu) < groups (%llu) — only %llu groups will be processed.\n",
                     threads, (unsigned long long)groups, threads);
    }

    uint32_t *d_in=nullptr, *d_out=nullptr;
    CUDA_OK(cudaMalloc(&d_in,  nbytes));
    CUDA_OK(cudaMalloc(&d_out, nbytes));
    CUDA_OK(cudaMemcpy(d_in, h_in_u32, nbytes, cudaMemcpyHostToDevice));



    // --- Kernel timing (CUDA events) ---
    cudaEvent_t ev_start, ev_stop;
    CUDA_OK(cudaEventCreate(&ev_start));
    CUDA_OK(cudaEventCreate(&ev_stop));


    // --- Warm Up ---

    for (int i=1; i<100; i++){
       aes128_bs32_full<MyKey><<<grid, block>>>(d_in, d_out, groups);
    }


    CUDA_OK(cudaEventRecord(ev_start));
    aes128_bs32_full<MyKey><<<grid, block>>>(d_in, d_out, groups);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaEventRecord(ev_stop));
    CUDA_OK(cudaEventSynchronize(ev_stop));

    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    // Copy back results
    CUDA_OK(cudaMemcpy(h_out_u32, d_out, nbytes, cudaMemcpyDeviceToHost));
    cudaFree(d_in);
    cudaFree(d_out);

    // --- Throughput accounting ---
    // Each *active* thread processes 32 blocks of 128 bits.
    const unsigned long long active_threads = (threads < groups) ? threads : groups;
    const unsigned long long blocks_total   = active_threads * 32ull;       // AES blocks (16B each)
    const unsigned long long bits_total     = blocks_total * 128ull;        // total bits encrypted

    const double seconds = (double)ms * 1e-3;
    const double bps     = (seconds > 0.0) ? (double)bits_total / seconds : 0.0;
    const double gbps    = bps / 1e9;

    std::printf(
        "KERNEL: %.3f ms | grid=(%u,%u,%u) block=(%u,%u,%u) | threads=%llu, groups=%llu, active=%llu\n",
        ms, grid.x,grid.y,grid.z, block.x,block.y,block.z,
        threads, (unsigned long long)groups, active_threads);
    std::printf(
        "WORK: blocks=%llu (32 per thread) = %.3f MiB payload | THROUGHPUT: %.2f Gbps\n",
        blocks_total,
        blocks_total * 16.0 / (1024.0*1024.0),
        gbps);

    return 0;
}



/**
 * @brief Host function to run bitsliced AES-128 and capture a specific stage.
 * 
 * @param h_in_u32 Host pointer to input bitsliced state.
 * @param h_out_u32 Host pointer to output bitsliced state.
 * @param groups Number of 128-slice groups.
 * @param grid CUDA grid dimensions.
 * @param block CUDA block dimensions.
 * @param target_stage The stage to dump (0..39).
 * @param h_dbg [out] Vector to store the captured debug stage data.
 * @return 0 on success, -1 on failure.
 */
extern "C" int run_aes_bs_full_stage(const uint32_t* h_in_u32,
                          uint32_t* h_out_u32,
                          unsigned long long groups,
                          dim3 grid, dim3 block,
                          int target_stage,
                          std::vector<uint32_t>& h_dbg)    // out: groups*128 u32
{
    const size_t N = (size_t)groups * 128ull;
    const size_t nbytes = N * sizeof(uint32_t);

    uint32_t *d_in=nullptr, *d_out=nullptr, *d_dbg=nullptr;
    cudaMalloc(&d_in,  nbytes);
    cudaMalloc(&d_out, nbytes);
    cudaMalloc(&d_dbg, (size_t)groups * 128ull * sizeof(uint32_t));
    cudaMemcpy(d_in, h_in_u32, nbytes, cudaMemcpyHostToDevice);

    aes128_bs32_full_debug<MyKey><<<grid, block>>>(d_in, d_out, groups, d_dbg, target_stage);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(e));
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_dbg);
        return -1;
    }

    cudaMemcpy(h_out_u32, d_out, nbytes, cudaMemcpyDeviceToHost);
    h_dbg.resize((size_t)groups * 128ull);
    cudaMemcpy(h_dbg.data(), d_dbg, h_dbg.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_dbg);
    return 0;
}


/**
 * @brief Host function to run bitsliced AES-128 and dump a specific stage to a provided buffer.
 * 
 * @param h_in_u32 Host pointer to input bitsliced state.
 * @param h_out_u32 Host pointer to output bitsliced state.
 * @param groups Number of 128-slice groups.
 * @param grid CUDA grid dimensions.
 * @param block CUDA block dimensions.
 * @param target_stage The stage to dump (0..39).
 * @param h_dbg_out [out] Host buffer to store the captured stage data.
 * @return 0 on success, -1 on failure.
 */
extern "C" int run_aes_bs_full_dump(const uint32_t* h_in_u32,
                                    uint32_t*       h_out_u32,
                                    unsigned long long groups,
                                    dim3 grid, dim3 block,
                                    int target_stage,              // which step to capture (0..39)
                                    uint32_t*       h_dbg_out)     // host buffer: groups*128 u32
{
    if (groups == 0) return 0;
    if (!h_in_u32 || !h_out_u32 || !h_dbg_out) {
        std::fprintf(stderr, "run_aes_bs_full_dump: null pointer argument\n");
        return -1;
    }
    if (target_stage < 0 || target_stage > 39) {
        std::fprintf(stderr, "run_aes_bs_full_dump: target_stage out of range [0,39]\n");
        return -1;
    }

    const size_t Nwords = (size_t)groups * 128ull;    // slices per group
    const size_t Nbytes = Nwords * sizeof(uint32_t);

    // (Optional) warn if grid*block < groups
    const unsigned long long threads =
        (unsigned long long)grid.x * block.x *
        (unsigned long long)grid.y * block.y *
        (unsigned long long)grid.z * block.z;
    if (threads < groups) {
        std::fprintf(stderr, "WARNING: threads (%llu) < groups (%llu)\n",
                     threads, (unsigned long long)groups);
    }

    uint32_t *d_in=nullptr, *d_out=nullptr, *d_dbg=nullptr;
    CUDA_OK(cudaMalloc(&d_in,  Nbytes));
    CUDA_OK(cudaMalloc(&d_out, Nbytes));
    CUDA_OK(cudaMalloc(&d_dbg, Nbytes));  // one stage per group -> groups*128 u32

    CUDA_OK(cudaMemcpy(d_in, h_in_u32, Nbytes, cudaMemcpyHostToDevice));

    // Launch kernel with debug buffer + target stage
    aes128_bs32_full_debug<MyKey><<<grid, block>>>(d_in, d_out, groups, d_dbg, target_stage);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    // Copy back both final output and the chosen stage dump
    CUDA_OK(cudaMemcpy(h_out_u32, d_out, Nbytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_dbg_out, d_dbg, Nbytes, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_dbg);
    return 0;
}
