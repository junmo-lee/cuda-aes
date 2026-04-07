#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>

__global__ void aes_bs_mix_only(const uint32_t*, uint32_t*, unsigned long long);

/**
 * @brief Reads the entire content of a file into a byte vector.
 * 
 * @param path The path to the file to read.
 * @param buf [out] The vector to store the file content.
 * @return true if reading is successful, false otherwise.
 */
static bool read_all(const std::string& path, std::vector<uint8_t>& buf) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    size_t sz = (size_t)f.tellg();
    f.seekg(0, std::beg);
    buf.resize(sz);
    f.read((char*)buf.data(), sz);
    return f.good();
}

/**
 * @brief Writes the entire content of a byte vector to a file.
 * 
 * @param path The path to the file to write.
 * @param buf The vector containing the data to write.
 * @return true if writing is successful, false otherwise.
 */
static bool write_all(const std::string& path, const std::vector<uint8_t>& buf) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write((const char*)buf.data(), buf.size());
    return f.good();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_slices_u32_le.bin> <output_slices_u32_le.bin> [block=256]\n", argv[0]);
        return 2;
    }
    std::string in_path  = argv[1];
    std::string out_path = argv[2];
    int block = (argc >= 4) ? atoi(argv[3]) : 256;

    std::vector<uint8_t> in_bytes;
    if (!read_all(in_path, in_bytes)) { fprintf(stderr, "read fail\n"); return 1; }
    if (in_bytes.size() % 4 != 0) { fprintf(stderr, "size mod 4 fail\n"); return 1; }
    size_t u32s = in_bytes.size() / 4;
    if (u32s % 128 != 0) { fprintf(stderr, "u32s mod 128 fail\n"); return 1; }
    unsigned long long groups = u32s / 128;

    std::vector<uint32_t> in_u32(u32s);
    for (size_t i = 0; i < u32s; ++i) {
        in_u32[i] = (uint32_t)in_bytes[4*i] | ((uint32_t)in_bytes[4*i+1] << 8) |
                    ((uint32_t)in_bytes[4*i+2] << 16) | ((uint32_t)in_bytes[4*i+3] << 24);
    }

    uint32_t *d_in=nullptr, *d_out=nullptr;
    cudaMalloc(&d_in,  in_u32.size()*sizeof(uint32_t));
    cudaMalloc(&d_out, in_u32.size()*sizeof(uint32_t));
    cudaMemcpy(d_in, in_u32.data(), in_u32.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);

    int grid = (int)((groups + block - 1) / block);
    aes_bs_mix_only<<<dim3(grid,1,1), dim3(block,1,1)>>>(d_in, d_out, groups);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); return 1; }

    std::vector<uint32_t> out_u32(u32s);
    cudaMemcpy(out_u32.data(), d_out, out_u32.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);

    std::vector<uint8_t> out_bytes(out_u32.size()*4);
    for (size_t i = 0; i < out_u32.size(); ++i) {
        uint32_t v = out_u32[i];
        out_bytes[4*i+0] = (uint8_t)(v & 0xFF);
        out_bytes[4*i+1] = (uint8_t)((v >> 8) & 0xFF);
        out_bytes[4*i+2] = (uint8_t)((v >> 16) & 0xFF);
        out_bytes[4*i+3] = (uint8_t)((v >> 24) & 0xFF);
    }
    if (!write_all(out_path, out_bytes)) { fprintf(stderr, "write fail\n"); return 1; }
    printf("Done. groups=%llu\n", groups);
    return 0;
}
