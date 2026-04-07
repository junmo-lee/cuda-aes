#pragma once

/**
 * @brief Simple tag type for compile-time key bytes.
 * 
 * This template struct holds 16 bytes of an AES key as a compile-time constant.
 * 
 * @tparam Bytes The 16 bytes of the key.
 */
template<unsigned char... Bytes>
struct StaticKey {
    /**
     * @brief The key data stored as a static constexpr array.
     */
    static constexpr unsigned char data[16] = {Bytes...};
};

/**
 * @brief Definition for the static constexpr data member of StaticKey.
 */
template<unsigned char... Bytes>
constexpr unsigned char StaticKey<Bytes...>::data[16];
