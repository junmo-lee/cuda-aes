#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <limits>
#include <cstring>

// Existing runner (final output)
extern "C" int run_aes_bs_full(const uint32_t*, uint32_t*, unsigned long long, dim3, dim3);

// NEW: stage dump runner (one stage per call)
extern "C" int run_aes_bs_full_dump(const uint32_t* h_in_u32,
                                    uint32_t*       h_out_u32,
                                    unsigned long long groups,
                                    dim3 grid, dim3 block,
                                    int target_stage,
                                    uint32_t* h_dbg_out /* groups*128 */);

static bool read_all(const std::string& path, std::vector<uint8_t>& buf) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "open fail: %s\n", path.c_str()); return false; }
    f.seekg(0, std::ios::end);
    std::streamoff end = f.tellg();
    if (end < 0) { fprintf(stderr, "tellg fail: %s\n", path.c_str()); return false; }
    size_t sz = static_cast<size_t>(end);
    f.seekg(0, std::ios::beg);
    buf.resize(sz);
    if (sz && !f.read(reinterpret_cast<char*>(buf.data()), sz)) {
        fprintf(stderr, "read bytes fail: %s\n", path.c_str());
        return false;
    }
    return true;
}
static bool write_all(const std::string& path, const std::vector<uint8_t>& buf) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) { fprintf(stderr, "open for write fail: %s\n", path.c_str()); return false; }
    if (!buf.empty() && !f.write(reinterpret_cast<const char*>(buf.data()), buf.size())) {
        fprintf(stderr, "write bytes fail: %s\n", path.c_str());
        return false;
    }
    return true;
}
static dim3 parse_xyz(const char* s) {
    unsigned int x=1,y=1,z=1; char sep;
    std::stringstream ss(s);
    ss >> x;
    if (ss.good() && (ss.peek()=='x' || ss.peek()=='X')) {
        ss >> sep >> y;
        if (ss.good() && (ss.peek()=='x' || ss.peek()=='X')) {
            ss >> sep >> z;
        }
    }
    return dim3(x,y,z);
}
static unsigned long long total_threads(dim3 grid, dim3 block) {
    return (unsigned long long)grid.x * block.x *
           (unsigned long long)grid.y * block.y *
           (unsigned long long)grid.z * block.z;
}

// Optional: pretty names for stage indices (0..39)
static const char* stage_name(int st, int* out_round=nullptr) {
    if (st == 0) { if(out_round)*out_round=0; return "r0_ark"; }
    if (1 <= st && st <= 36) {
        int r = 1 + (st-1)/4;
        int k = (st-1)%4; // 0:sb,1:sr,2:mc,3:ark
        if (out_round) *out_round = r;
        switch(k){
            case 0: return (std::string("r")+std::to_string(r)+"_sb").c_str();
            case 1: return (std::string("r")+std::to_string(r)+"_sr").c_str();
            case 2: return (std::string("r")+std::to_string(r)+"_mc").c_str();
            default:return (std::string("r")+std::to_string(r)+"_ark").c_str();
        }
    }
    if (st == 37) { if(out_round)*out_round=10; return "r10_sb"; }
    if (st == 38) { if(out_round)*out_round=10; return "r10_sr"; }
    if (st == 39) { if(out_round)*out_round=10; return "r10_ark"; }
    return "stage";
}

static void usage(const char* prog) {
    fprintf(stderr,
      "Usage:\n"
      "  %s <input_slices.bin> <final_out.bin> <grid> <block>\n"
      "     [--dump-stage <idx> <dump_file>]                   # dump one stage (0..39)\n"
      "     [--dump-all <dir>]                                 # dump all stages into dir\n"
      "\n"
      "Examples:\n"
      "  %s inputs/slices_u32_le.bin outputs/run_full_slices_u32_le.bin 32x1x1 128x1x1\n"
      "  %s inputs/slices_u32_le.bin outputs/run_full_slices_u32_le.bin 32x1x1 128x1x1 --dump-stage 0 outputs/dump_r0_ark.bin\n"
      "  %s inputs/slices_u32_le.bin outputs/run_full_slices_u32_le.bin 32x1x1 128x1x1 --dump-all outputs/stages\n",
      prog, prog, prog, prog);
}

int main(int argc, char** argv) {
    if (argc < 5) { usage(argv[0]); return 2; }

    std::string in_path  = argv[1];
    std::string out_path = argv[2];
    dim3 grid  = parse_xyz(argv[3]);
    dim3 block = parse_xyz(argv[4]);

    // Parse optional args
    bool do_dump_one = false, do_dump_all = false;
    int dump_stage_idx = -1;
    std::string dump_one_path, dump_dir;

    for (int i = 5; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump-stage") == 0 && i+2 < argc) {
            do_dump_one = true;
            dump_stage_idx = std::stoi(argv[i+1]);
            dump_one_path = argv[i+2];
            i += 2;
        } else if (std::strcmp(argv[i], "--dump-all") == 0 && i+1 < argc) {
            do_dump_all = true;
            dump_dir = argv[i+1];
            i += 1;
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    if (do_dump_one && (dump_stage_idx < 0 || dump_stage_idx > 39)) {
        fprintf(stderr, "--dump-stage index must be in [0,39]\n");
        return 2;
    }

    // Read input bytes (little-endian u32)
    std::vector<uint8_t> in_bytes;
    if (!read_all(in_path, in_bytes)) return 1;

    if (in_bytes.size() % 4 != 0) {
        fprintf(stderr, "size mod 4 fail: %zu\n", in_bytes.size());
        return 1;
    }
    const size_t u32s = in_bytes.size() / 4;
    if (u32s % 128 != 0) {
        fprintf(stderr, "u32s mod 128 fail: %zu\n", u32s);
        return 1;
    }
    const unsigned long long groups = static_cast<unsigned long long>(u32s / 128);

    // Optional: warn if launch undersubscribes groups
    const unsigned long long threads = total_threads(grid, block);
    if (threads < groups) {
        fprintf(stderr,
            "WARNING: threads (%llu) < groups (%llu); some groups won't be processed.\n",
            threads, groups);
    }

    // Convert to u32 (LE)
    std::vector<uint32_t> in_u32(u32s);
    for (size_t i = 0; i < u32s; ++i) {
        const size_t off = i*4;
        in_u32[i] = (uint32_t)in_bytes[off+0]
                  | ((uint32_t)in_bytes[off+1] << 8)
                  | ((uint32_t)in_bytes[off+2] << 16)
                  | ((uint32_t)in_bytes[off+3] << 24);
    }

    std::vector<uint32_t> out_u32(u32s, 0);

    // Always produce the final output file:
    if (run_aes_bs_full(in_u32.data(), out_u32.data(), groups, grid, block) != 0) {
        fprintf(stderr, "run_aes_bs_full failed\n");
        return 1;
    }
    // Write final (LE)
    {
        std::vector<uint8_t> out_bytes(out_u32.size()*4);
        for (size_t i = 0; i < out_u32.size(); ++i) {
            const uint32_t v = out_u32[i];
            const size_t off = i*4;
            out_bytes[off+0] = (uint8_t)(v & 0xFF);
            out_bytes[off+1] = (uint8_t)((v >> 8) & 0xFF);
            out_bytes[off+2] = (uint8_t)((v >> 16) & 0xFF);
            out_bytes[off+3] = (uint8_t)((v >> 24) & 0xFF);
        }
        if (!write_all(out_path, out_bytes)) return 1;
    }

    // Dump ONE stage if requested
    if (do_dump_one) {
        std::vector<uint32_t> dbg(groups * 128ull, 0);
        if (run_aes_bs_full_dump(in_u32.data(), out_u32.data(), groups, grid, block,
                                 dump_stage_idx, dbg.data()) != 0) {
            fprintf(stderr, "run_aes_bs_full_dump(stage=%d) failed\n", dump_stage_idx);
            return 1;
        }
        std::vector<uint8_t> bytes(dbg.size()*4);
        for (size_t i=0;i<dbg.size();++i) {
            uint32_t v = dbg[i];
            bytes[4*i+0]=(uint8_t)(v);
            bytes[4*i+1]=(uint8_t)(v>>8);
            bytes[4*i+2]=(uint8_t)(v>>16);
            bytes[4*i+3]=(uint8_t)(v>>24);
        }
        if (!write_all(dump_one_path, bytes)) return 1;
        printf("Dumped stage %d -> %s\n", dump_stage_idx, dump_one_path.c_str());
    }

    // Dump ALL stages (0..39) if requested
    if (do_dump_all) {
        // ensure dir exists (best-effort; ignore errors)
        #ifdef _WIN32
          _mkdir(dump_dir.c_str());
        #else
          std::string mkdir_cmd = "mkdir -p \"" + dump_dir + "\"";
          (void)std::system(mkdir_cmd.c_str());
        #endif
        std::vector<uint32_t> dbg(groups * 128ull, 0);
        char fname[512];
        for (int st = 0; st <= 39; ++st) {
            if (run_aes_bs_full_dump(in_u32.data(), out_u32.data(), groups, grid, block,
                                     st, dbg.data()) != 0) {
                fprintf(stderr, "run_aes_bs_full_dump(stage=%d) failed\n", st);
                return 1;
            }
            // file name: try to add human-friendly tag
            int r=-1; (void)r;
            const char* tag = stage_name(st, &r);
            // Fall back safely if tag uses temporary string; format explicitly:
            std::snprintf(fname, sizeof(fname), "%s/dump_stage_%02d.bin", dump_dir.c_str(), st);

            // write LE
            std::vector<uint8_t> bytes(dbg.size()*4);
            for (size_t i=0;i<dbg.size();++i) {
                uint32_t v = dbg[i];
                bytes[4*i+0]=(uint8_t)(v);
                bytes[4*i+1]=(uint8_t)(v>>8);
                bytes[4*i+2]=(uint8_t)(v>>16);
                bytes[4*i+3]=(uint8_t)(v>>24);
            }
            if (!write_all(fname, bytes)) { fprintf(stderr, "write fail: %s\n", fname); return 1; }
            printf("Dumped stage %2d -> %s\n", st, fname);
        }
        printf("All stages dumped to: %s\n", dump_dir.c_str());
    }

    printf("OK: groups=%llu, grid=(%u,%u,%u) block=(%u,%u,%u)\n",
           groups, grid.x,grid.y,grid.z, block.x,block.y,block.z);
    return 0;
}
