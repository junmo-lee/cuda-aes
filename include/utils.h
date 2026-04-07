#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <sstream>
#include <iomanip>

/**
 * @brief 8-bit unsigned integer type.
 */
typedef unsigned char      u8;
/**
 * @brief 16-bit unsigned integer type.
 */
typedef unsigned short     u16;
/**
 * @brief 32-bit unsigned integer type.
 */
typedef unsigned int       u32;
/**
 * @brief 64-bit unsigned integer type.
 */
typedef unsigned long long u64;

// ─── 128-bit Key ─────────────────────────────────────────────────────────────
/**
 * @brief Represents a 128-bit AES key.
 * 
 * Internally stored as four 32-bit words in big-endian order.
 *   word[0] = most significant word  (rk0)
 *   word[3] = least significant word (rk3)
 */
struct Key128 {
    u32 w[4];  ///< Internal storage for the four 32-bit words (w[0] = MSW, w[3] = LSW).

    /**
     * @brief Default constructor, initializes the key to all zeros.
     */
    Key128() { w[0] = w[1] = w[2] = w[3] = 0; }
    
    /**
     * @brief Constructor with individual word initialization.
     * @param a Most significant word.
     * @param b Second most significant word.
     * @param c Second least significant word.
     * @param d Least significant word.
     */
    Key128(u32 a, u32 b, u32 c, u32 d) { w[0]=a; w[1]=b; w[2]=c; w[3]=d; }

    // ── arithmetic ─────────────────────────────────────────────────────────
    /**
     * @brief Addition operator for 128-bit key and a 64-bit value.
     * @param val The 64-bit value to add.
     * @return A new Key128 representing the sum.
     */
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
    
    /**
     * @brief Addition assignment operator.
     * @param val The 64-bit value to add.
     * @return Reference to the updated Key128.
     */
    Key128& operator+=(u64 val) { *this = *this + val; return *this; }

    /**
     * @brief Equality comparison operator.
     * @param o The other Key128 to compare with.
     * @return true if both keys are equal, false otherwise.
     */
    bool operator==(const Key128& o) const {
        return w[0]==o.w[0] && w[1]==o.w[1] && w[2]==o.w[2] && w[3]==o.w[3];
    }
    
    /**
     * @brief Less-than comparison operator for key ranges.
     * @param o The other Key128 to compare with.
     * @return true if this key is less than the other key.
     */
    bool operator<(const Key128& o) const {
        for (int i=0;i<4;i++) {
            if (w[i]!=o.w[i]) return w[i]<o.w[i];
        }
        return false;
    }

    // ── I/O ────────────────────────────────────────────────────────────────
    /**
     * @brief Converts the key to a hexadecimal string representation.
     * @return Hexadecimal string of the 128-bit key.
     */
    std::string toHex() const {
        std::ostringstream ss;
        ss << std::hex << std::setfill('0');
        for (int i=0;i<4;i++) ss << std::setw(8) << w[i];
        return ss.str();
    }

    /**
     * @brief Fills a 16-byte array with the key's big-endian representation.
     * @param out Pointer to the output 16-byte array.
     */
    void toBytes(u8 out[16]) const {
        for (int i=0;i<4;i++) {
            out[i*4+0] = (u8)(w[i] >> 24);
            out[i*4+1] = (u8)(w[i] >> 16);
            out[i*4+2] = (u8)(w[i] >>  8);
            out[i*4+3] = (u8)(w[i]      );
        }
    }

    /**
     * @brief Creates a Key128 from a 16-byte big-endian representation.
     * @param b Pointer to the 16-byte input array.
     * @return A Key128 object initialized from the bytes.
     */
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
/**
 * @brief Represents a 128-bit (16-byte) AES block.
 */
struct Block128 {
    u8 data[16]; ///< Internal storage for the 16 bytes.
    
    /**
     * @brief Default constructor, initializes the block to all zeros.
     */
    Block128()               { memset(data, 0, 16); }
    
    /**
     * @brief Explicit constructor from a 16-byte array.
     * @param p Pointer to the 16-byte source array.
     */
    explicit Block128(const u8* p) { memcpy(data, p, 16); }
    
    /**
     * @brief Equality comparison operator.
     * @param o The other Block128 to compare with.
     * @return true if both blocks are identical, false otherwise.
     */
    bool operator==(const Block128& o) const { return !memcmp(data, o.data, 16); }

    /**
     * @brief Views the block as four 32-bit words (big-endian).
     * @param out Pointer to the output 4-word array.
     */
    void toWords(u32 out[4]) const {
        for (int i=0;i<4;i++) {
            out[i] = ((u32)data[i*4+0]<<24) | ((u32)data[i*4+1]<<16)
                   | ((u32)data[i*4+2]<< 8) |  (u32)data[i*4+3];
        }
    }
};

// ─── Work assignment (MPI payload) ───────────────────────────────────────────
/**
 * @brief Represents a work assignment sent from master to worker via MPI.
 */
struct WorkAssignment {
    u8  ciphertext[16]; ///< The 128-bit ciphertext to be broken.
    u8  plaintext[16];  ///< The 128-bit known plaintext for verification.
    u32 key_start[4];   ///< Inclusive start of the key range.
    u32 key_end[4];     ///< Exclusive end of the key range (all-zero = STOP signal).
};

// ─── Result from a worker ────────────────────────────────────────────────────
/**
 * @brief Represents the result of a work assignment sent back to the master.
 */
struct WorkResult {
    int  rank;          ///< MPI rank of the worker.
    int  found;         ///< Flag indicating if the key was found (1 = found, 0 = not found).
    u32  key[4];        ///< The found 128-bit key (valid only if found == 1).
};

// ─── Progress report ─────────────────────────────────────────────────────────
/**
 * @brief Represents a periodic progress report from a worker.
 */
struct ProgressReport {
    int    rank;             ///< MPI rank of the worker.
    u64    keys_tried;       ///< Total number of keys tried in the last interval.
    double keys_per_second;  ///< Current performance in keys per second.
};

// ─── MPI tags ────────────────────────────────────────────────────────────────
/**
 * @brief MPI message tags used for communication between master and workers.
 */
enum MpiTag : int {
    TAG_WORK     = 1, ///< Work assignment message.
    TAG_RESULT   = 2, ///< Work result message.
    TAG_PROGRESS = 3, ///< Progress report message.
};
