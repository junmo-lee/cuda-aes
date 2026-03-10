#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <sstream>
#include <iomanip>

typedef unsigned char      u8;
typedef unsigned short     u16;
typedef unsigned int       u32;
typedef unsigned long long u64;

// ─── 128-bit Key ─────────────────────────────────────────────────────────────
// Internally stored as four 32-bit words (big-endian order, same as reference).
//   word[0] = most significant word  (rk0)
//   word[3] = least significant word (rk3)
struct Key128 {
    u32 w[4];  // w[0] = MSW, w[3] = LSW

    Key128() { w[0] = w[1] = w[2] = w[3] = 0; }
    Key128(u32 a, u32 b, u32 c, u32 d) { w[0]=a; w[1]=b; w[2]=c; w[3]=d; }

    // ── arithmetic ─────────────────────────────────────────────────────────
    Key128 operator+(u64 val) const {
        Key128 r = *this;
        u64 lo = ((u64)r.w[2] << 32) | r.w[3];
        u64 hi = ((u64)r.w[0] << 32) | r.w[1];
        lo += val;
        if (lo < val) hi++;          // carry
        r.w[0] = (u32)(hi >> 32);
        r.w[1] = (u32)(hi & 0xFFFFFFFFULL);
        r.w[2] = (u32)(lo >> 32);
        r.w[3] = (u32)(lo & 0xFFFFFFFFULL);
        return r;
    }
    Key128& operator+=(u64 val) { *this = *this + val; return *this; }

    bool operator==(const Key128& o) const {
        return w[0]==o.w[0] && w[1]==o.w[1] && w[2]==o.w[2] && w[3]==o.w[3];
    }
    bool operator<(const Key128& o) const {
        for (int i=0;i<4;i++) {
            if (w[i]!=o.w[i]) return w[i]<o.w[i];
        }
        return false;
    }

    // ── I/O ────────────────────────────────────────────────────────────────
    std::string toHex() const {
        std::ostringstream ss;
        ss << std::hex << std::setfill('0');
        for (int i=0;i<4;i++) ss << std::setw(8) << w[i];
        return ss.str();
    }

    // fill 16-byte array (big-endian)
    void toBytes(u8 out[16]) const {
        for (int i=0;i<4;i++) {
            out[i*4+0] = (u8)(w[i] >> 24);
            out[i*4+1] = (u8)(w[i] >> 16);
            out[i*4+2] = (u8)(w[i] >>  8);
            out[i*4+3] = (u8)(w[i]      );
        }
    }

    static Key128 fromBytes(const u8 b[16]) {
        Key128 k;
        for (int i=0;i<4;i++) {
            k.w[i] = ((u32)b[i*4+0]<<24) | ((u32)b[i*4+1]<<16)
                   | ((u32)b[i*4+2]<< 8) |  (u32)b[i*4+3];
        }
        return k;
    }
};

// ─── 16-byte block ───────────────────────────────────────────────────────────
struct Block128 {
    u8 data[16];
    Block128()               { memset(data, 0, 16); }
    explicit Block128(const u8* p) { memcpy(data, p, 16); }
    bool operator==(const Block128& o) const { return !memcmp(data, o.data, 16); }

    // view as 4 u32 words (big-endian)
    void toWords(u32 out[4]) const {
        for (int i=0;i<4;i++) {
            out[i] = ((u32)data[i*4+0]<<24) | ((u32)data[i*4+1]<<16)
                   | ((u32)data[i*4+2]<< 8) |  (u32)data[i*4+3];
        }
    }
};

// ─── Work assignment (MPI payload) ───────────────────────────────────────────
struct WorkAssignment {
    u8  ciphertext[16];
    u8  plaintext[16];
    u32 key_start[4];   // inclusive
    u32 key_end[4];     // exclusive  (all-zero = STOP signal)
};

// ─── Result from a worker ────────────────────────────────────────────────────
struct WorkResult {
    int  rank;
    int  found;         // 1 = key found, 0 = not found in range
    u32  key[4];        // valid when found==1
};

// ─── Progress report ─────────────────────────────────────────────────────────
struct ProgressReport {
    int    rank;
    u64    keys_tried;
    double keys_per_second;
};

// ─── MPI tags ────────────────────────────────────────────────────────────────
enum MpiTag : int {
    TAG_WORK     = 1,
    TAG_RESULT   = 2,
    TAG_PROGRESS = 3,
};
