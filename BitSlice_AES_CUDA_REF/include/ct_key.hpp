#pragma once
// Simple tag type for compile-time key bytes
template<unsigned char... Bytes>
struct StaticKey {
    static constexpr unsigned char data[16] = {Bytes...};
};
template<unsigned char... Bytes>
constexpr unsigned char StaticKey<Bytes...>::data[16];
