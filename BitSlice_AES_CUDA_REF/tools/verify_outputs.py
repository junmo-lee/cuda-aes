#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, json, hashlib
from typing import List, Tuple
from dataclasses import dataclass

# You already have this helper module alongside your repo
from bitslice32 import read_slices_u32_le

# ---------------- AES reference (PyCryptodome if available, else pure-Python) ----------------
try:
    from Crypto.Cipher import AES as _AES

    def aes128_ecb_encrypt(key: bytes, blocks: List[bytes]) -> List[bytes]:
        cipher = _AES.new(key, _AES.MODE_ECB)
        return [cipher.encrypt(b) for b in blocks]
except Exception:
    # Minimal AES-128 (ECB) reference
    SBOX = [
      0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
      0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
      0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
      0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
      0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
      0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
      0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
      0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
      0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
      0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
      0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
      0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
      0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
      0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
      0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
      0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
    ]
    RCON = [0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36]
    def _rotl8(w): return ((w<<8)&0xFFFFFFFF) | (w>>24)
    def _subword(w):
        return ((SBOX[(w>>24)&0xFF]<<24) | (SBOX[(w>>16)&0xFF]<<16) |
                (SBOX[(w>>8)&0xFF]<<8) | (SBOX[w&0xFF]))
    def _expand(key: bytes) -> bytes:
        assert len(key)==16
        w = [0]*44
        for i in range(4):
            w[i] = (key[4*i]<<24)|(key[4*i+1]<<16)|(key[4*i+2]<<8)|(key[4*i+3])
        rc_idx = 1
        for i in range(4,44):
            temp = w[i-1]
            if i%4==0:
                temp = _subword(_rotl8(temp)) ^ (RCON[rc_idx]<<24)
                rc_idx += 1
            w[i] = (w[i-4] ^ temp) & 0xFFFFFFFF
        out = bytearray(176)
        for r in range(11):
            for j in range(4):
                v = w[4*r+j]
                out[16*r+4*j+0]=(v>>24)&0xFF
                out[16*r+4*j+1]=(v>>16)&0xFF
                out[16*r+4*j+2]=(v>>8)&0xFF
                out[16*r+4*j+3]=(v)&0xFF
        return bytes(out)
    def _xtime(a): return ((a<<1)&0xFF) ^ (0x1B if (a&0x80) else 0x00)
    def _mixcol(a,b,c,d):
        return (
            _xtime(a)^b^_xtime(b)^c^d,
            a^_xtime(b)^c^_xtime(c)^d,
            a^b^_xtime(c)^d^_xtime(d),
            _xtime(a)^a^b^c^_xtime(d)
        )
    def _shiftrows(s):
        return bytes([
            s[0],s[5],s[10],s[15],
            s[4],s[9],s[14],s[3],
            s[8],s[13],s[2],s[7],
            s[12],s[1],s[6],s[11]
        ])
    def _subbytes(s): return bytes([SBOX[x] for x in s])
    def aes128_ecb_encrypt(key: bytes, blocks: List[bytes]) -> List[bytes]:
        rk = _expand(key)
        out = []
        for b in blocks:
            s = bytes([b[i]^rk[i] for i in range(16)])
            for r in range(1,10):
                s = _subbytes(s)
                s = _shiftrows(s)
                t = bytearray(16)
                for c in range(4):
                    a,b2,c2,d = s[4*c:4*c+4]
                    x0,x1,x2,x3 = _mixcol(a,b2,c2,d)
                    t[4*c+0]=x0; t[4*c+1]=x1; t[4*c+2]=x2; t[4*c+3]=x3
                s = bytes([t[i]^rk[16*r+i] for i in range(16)])
            s = _subbytes(s)
            s = _shiftrows(s)
            s = bytes([s[i]^rk[160+i] for i in range(16)])
            out.append(s)
        return out

# ---------------- utilities ----------------
@dataclass
class DiffResult:
    ok: bool
    first_block: int
    first_byte: int
    a_hex: str
    b_hex: str

def compare_blocks(a_blocks: List[bytes], b_blocks: List[bytes], limit:int=4) -> DiffResult:
    n = min(len(a_blocks), len(b_blocks))
    first_block = -1
    first_byte = -1
    for i in range(n):
        if a_blocks[i] != b_blocks[i]:
            first_block = i
            for j, (aa, bb) in enumerate(zip(a_blocks[i], b_blocks[i])):
                if aa != bb:
                    first_byte = j
                    break
            break
    ok = (first_block == -1)
    a_hex = b_hex = ""
    if not ok:
        a_hex = a_blocks[first_block].hex()
        b_hex = b_blocks[first_block].hex()
    return DiffResult(ok, first_block, first_byte, a_hex, b_hex)

def hamming_bytes(a: bytes, b: bytes) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# ---------------- layout + transforms ----------------
def transpose16(b: bytes) -> bytes:
    """Transpose a 4x4 AES state (column-major). i -> (i%4)*4 + (i//4)."""
    assert len(b) == 16
    out = bytearray(16)
    for i in range(16):
        r, c = i % 4, i // 4
        j = r * 4 + c
        out[j] = b[i]
    return bytes(out)

def apply_block_transform(blocks: List[bytes], transform: str) -> List[bytes]:
    if transform == "id":
        return blocks
    elif transform == "transpose":
        return [transpose16(x) for x in blocks]
    else:
        raise ValueError("bad transform")

def unpack_group_32x128_layout(group_slices, order: str, bitorder: str, lane_order: str) -> List[bytes]:
    """
    order:     'byte_major' -> slice = byte*8 + bit
               'bit_major'  -> slice = bit*16 + byte
    bitorder:  'lsb' (bit0..7) or 'msb' (bit7..0)
    lane_order:'fwd' (0..31) or 'rev' (31..0)
    """
    assert len(group_slices) == 128
    out = []
    lane_index = range(32) if lane_order == "fwd" else range(31, -1, -1)

    def idx(byte_idx, bit):
        return byte_idx*8 + bit if order == "byte_major" else bit*16 + byte_idx

    for lane in lane_index:
        b = bytearray(16)
        for byte_idx in range(16):
            v = 0
            for bit in range(8):
                s = group_slices[idx(byte_idx, bit)]
                if (s >> lane) & 1:
                    pos = bit if bitorder == "lsb" else (7 - bit)
                    v |= (1 << pos)
            b[byte_idx] = v
        out.append(bytes(b))
    return out

def unpack_all_layout(groups: int, slices: List[int], order: str, bitorder: str, lane_order: str) -> List[bytes]:
    out = []
    for g in range(groups):
        gs = slices[128*g:128*(g+1)]
        out.extend(unpack_group_32x128_layout(gs, order, bitorder, lane_order))
    return out

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Verify AES outputs vs Python AES and SHA256.")
    ap.add_argument("--meta", required=True, help="meta.json from inputs run")
    ap.add_argument("--slices_out", required=True, help="slices_u32_le.bin produced by CUDA")
    ap.add_argument("--keyhex", default="2b7e151628aed2a6abf7158809cf4f3c",
                    help="AES-128 key hex (must match compile-time key)")
    ap.add_argument("--bitorder", choices=["lsb","msb","auto"], default="auto",
                    help="Interpretation of bit-slices when unpacking (default: auto)")
    ap.add_argument("--assume_identity", action="store_true",
                    help="Assume kernel output is identity copy; compare to plaintexts.")
    ap.add_argument("--show", type=int, default=4, help="show first N blocks when diffing")
    args = ap.parse_args()

    # --- Load meta & plaintexts ---
    meta = json.load(open(args.meta, "r"))
    pt_path_bin = os.path.join(os.path.dirname(args.meta), "plaintexts.bin")
    pts = open(pt_path_bin, "rb").read()
    blocks = [pts[i:i+16] for i in range(0, len(pts), 16)]

    # --- Reference AES (Python) ---
    key = bytes.fromhex(args.keyhex)
    ref_ct_blocks = aes128_ecb_encrypt(key, blocks)

    # --- Read CUDA slices ---
    slices = read_slices_u32_le(args.slices_out)
    if len(slices) % 128 != 0:
        raise SystemExit(f"slices_out length invalid: {len(slices)} (not multiple of 128)")
    groups = len(slices) // 128
    if groups * 32 != len(blocks):
        # Not fatal, but warn if your meta/plaintexts length doesn't match groups*32
        print(f"NOTE: plaintext blocks = {len(blocks)}, CUDA groups*32 = {groups*32}")

    # --- Unpack with auto layout search (8 combos × optional transpose) ---
    if args.bitorder == "auto":
        combos = [
            ("byte_major","lsb","fwd"),
            ("byte_major","lsb","rev"),
            ("byte_major","msb","fwd"),
            ("byte_major","msb","rev"),
            ("bit_major","lsb","fwd"),
            ("bit_major","lsb","rev"),
            ("bit_major","msb","fwd"),
            ("bit_major","msb","rev"),
        ]
        transforms = ["id", "transpose"]

        print("== AUTO layout check ==")
        best = None
        best_score = 1 << 60
        found = False

        for order, bitord, laneord in combos:
            blocks_try = unpack_all_layout(groups, slices, order, bitord, laneord)
            for tf in transforms:
                ref_try = apply_block_transform(ref_ct_blocks, tf)
                diff = compare_blocks(blocks_try, ref_try, args.show)
                if diff.ok:
                    chosen = f"{order}/{bitord}/{laneord}/{tf}"
                    print(f"Match: {chosen}")
                    cuda_blocks = blocks_try
                    ref_ct_blocks = ref_try  # align to transform
                    found = True
                    break
                else:
                    # score: hamming on first differing block if available, else first 16 bytes
                    if diff.first_block >= 0 and diff.a_hex and diff.b_hex:
                        score = hamming_bytes(bytes.fromhex(diff.a_hex), bytes.fromhex(diff.b_hex))
                    else:
                        score = hamming_bytes(b"".join(blocks_try)[:16], b"".join(ref_try)[:16])
                    if score < best_score:
                        best_score = score
                        best = (order, bitord, laneord, tf, diff)
            if found:
                break

        if not found:
            print("No exact match in 16 combos.")
            if best:
                order, bitord, laneord, tf, diff = best
                print(f"Closest: {order}/{bitord}/{laneord}/{tf}")
                if diff.first_block >= 0:
                    print(f"  first mismatch at block {diff.first_block}, byte {diff.first_byte}")
                    print(f"  CUDA block hex: {diff.a_hex}")
                    print(f"  REF  block hex: {diff.b_hex}")
            # fall back so the script continues
            cuda_blocks = unpack_all_layout(groups, slices, "byte_major", "lsb", "fwd")
            chosen = "byte_major/lsb/fwd/id"
        else:
            print(f"Chosen layout = {chosen}")

    else:
        # Explicit bit order path: assume byte_major/fwd lanes, no transpose
        cuda_blocks = unpack_all_layout(groups, slices, "byte_major", args.bitorder, "fwd")

    # --- Optional identity check (for packing/unpacking debug) ---
    if args.assume_identity:
        id_diff = compare_blocks(cuda_blocks, blocks, args.show)
        print("== IDENTITY check (CUDA vs plaintexts) ==")
        print("Match? ", "YES" if id_diff.ok else "NO")
        if not id_diff.ok:
            print(f"  first mismatch at block {id_diff.first_block}, byte {id_diff.first_byte}")
            print(f"  CUDA block hex: {id_diff.a_hex}")
            print(f"  PT   block hex: {id_diff.b_hex}")

    # --- Write side-by-side binaries for inspection ---
    out_dir = os.path.dirname(args.slices_out)
    os.makedirs(out_dir, exist_ok=True)
    cuda_ct = b"".join(cuda_blocks)
    ref_ct  = b"".join(ref_ct_blocks)

    with open(os.path.join(out_dir, "ciphertexts_from_cuda.bin"), "wb") as f:
        f.write(cuda_ct)
    with open(os.path.join(out_dir, "ciphertexts_from_python.bin"), "wb") as f:
        f.write(ref_ct)

    # --- SHA summaries ---
    print("SHA256 CUDA : ", sha256_hex(cuda_ct))
    print("SHA256 PY   : ", sha256_hex(ref_ct))
    print("MATCH?      : ", "YES" if cuda_ct == ref_ct else "NO")

if __name__ == "__main__":
    main()
