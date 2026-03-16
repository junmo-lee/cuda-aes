#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, json, hashlib, struct
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Your helper to read CUDA-written slices (uint32 little-endian)
from bitslice32 import read_slices_u32_le

# ================= AES reference (PyCryptodome if available; else pure-Python) =================
try:
    from Crypto.Cipher import AES as _AES
    HAVE_PYCRYPTO = True
except Exception:
    HAVE_PYCRYPTO = False

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
def _shiftrows(s: bytes) -> bytes:
    # Column-major state: state[r][c] = s[4*c + r]
    return bytes([
        s[0],s[5],s[10],s[15],
        s[4],s[9],s[14],s[3],
        s[8],s[13],s[2],s[7],
        s[12],s[1],s[6],s[11]
    ])
def _subbytes(s: bytes) -> bytes:
    return bytes([SBOX[x] for x in s])

def aes128_ecb_encrypt(key: bytes, blocks: List[bytes]) -> List[bytes]:
    if HAVE_PYCRYPTO:
        cipher = _AES.new(key, _AES.MODE_ECB)
        return [cipher.encrypt(b) for b in blocks]
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

# ================= utilities =================
@dataclass
class Diff:
    ok: bool
    block: int
    byte: int
    got_hex: str
    exp_hex: str

def compare_blocks(a: List[bytes], b: List[bytes]) -> Diff:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            for j,(x,y) in enumerate(zip(a[i], b[i])):
                if x != y:
                    return Diff(False, i, j, a[i].hex(), b[i].hex())
    return Diff(True, -1, -1, "", "")

def sha256_hex(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()

# ================= bitslice pack/unpack & layout =================
Layout = Tuple[str,str,str]  # (order, bitorder, lane_order)

def pack_group(blocks32: List[bytes], order: str, bitorder: str, lane_order: str) -> List[int]:
    assert len(blocks32) == 32
    slices = [0]*128
    lane_idx = range(32) if lane_order=="fwd" else range(31,-1,-1)
    for byte_idx in range(16):
        for bit in range(8):
            idx = byte_idx*8 + bit if order=="byte_major" else bit*16 + byte_idx
            w = 0
            for lane_pos, lane in enumerate(lane_idx):
                b = blocks32[lane][byte_idx]
                k = bit if bitorder=="lsb" else (7-bit)
                if (b >> k) & 1:
                    w |= (1 << lane_pos)
            slices[idx] = w
    return slices

def unpack_group(slices128: List[int], order: str, bitorder: str, lane_order: str) -> List[bytes]:
    assert len(slices128) == 128
    blocks = [bytearray(16) for _ in range(32)]
    lane_idx = range(32) if lane_order=="fwd" else range(31,-1,-1)
    for byte_idx in range(16):
        for bit in range(8):
            idx = byte_idx*8 + bit if order=="byte_major" else bit*16 + byte_idx
            w = slices128[idx]
            for lane_pos, lane in enumerate(lane_idx):
                if (w >> lane_pos) & 1:
                    k = bit if bitorder=="lsb" else (7-bit)
                    blocks[lane][byte_idx] |= (1 << k)
    return [bytes(b) for b in blocks]

def detect_layout_from_inputs(pt_blocks: List[bytes], input_slices_u32: List[int]) -> Optional[Layout]:
    combos: List[Layout] = [
        ("byte_major","lsb","fwd"), ("byte_major","lsb","rev"),
        ("byte_major","msb","fwd"), ("byte_major","msb","rev"),
        ("bit_major","lsb","fwd"),  ("bit_major","lsb","rev"),
        ("bit_major","msb","fwd"),  ("bit_major","msb","rev"),
    ]
    groups = len(input_slices_u32) // 128
    for order, bitorder, lane in combos:
        guess: List[int] = []
        for g in range(groups):
            guess.extend(pack_group(pt_blocks[g*32:(g+1)*32], order, bitorder, lane))
        if guess == input_slices_u32:
            return (order, bitorder, lane)
    return None

# ================= round-by-round expected states =================
def aes128_round_states(key: bytes, blocks: List[bytes]):
    """
    Returns dicts of lists-of-blocks for each round/phase:
      ark[0..10], sb[1..10], sr[1..10], mc[1..9]
    """
    rk = _expand(key)
    states_ark = {r: [] for r in range(0, 11)}
    states_sb  = {r: [] for r in range(1, 11)}
    states_sr  = {r: [] for r in range(1, 11)}
    states_mc  = {r: [] for r in range(1, 10)}  # 1..9

    for b in blocks:
        # Round 0 ARK
        s = bytes([b[i]^rk[i] for i in range(16)])
        states_ark[0].append(s)
        # Rounds 1..9
        for r in range(1,10):
            s = _subbytes(s);     states_sb[r].append(s)
            s = _shiftrows(s);    states_sr[r].append(s)
            t = bytearray(16)
            for c in range(4):
                a,b2,c2,d = s[4*c:4*c+4]
                x0,x1,x2,x3 = _mixcol(a,b2,c2,d)
                t[4*c+0]=x0; t[4*c+1]=x1; t[4*c+2]=x2; t[4*c+3]=x3
            s = bytes(t);          states_mc[r].append(s)
            s = bytes([s[i]^rk[16*r+i] for i in range(16)])
            states_ark[r].append(s)
        # Round 10
        s = _subbytes(s);     states_sb[10].append(s)
        s = _shiftrows(s);    states_sr[10].append(s)
        s = bytes([s[i]^rk[160+i] for i in range(16)])
        states_ark[10].append(s)

    return states_ark, states_sb, states_sr, states_mc

def expected_blocks_for(phase: str, rnd: int,
                        pt_blocks: List[bytes], key: bytes) -> List[bytes]:
    """
    phase in {"final","ct","ark","sb","sr","mc"}; rnd=0..10 as applicable.
    """
    if phase in ("final","ct"):
        return aes128_ecb_encrypt(key, pt_blocks)
    ark, sb, sr, mc = aes128_round_states(key, pt_blocks)
    if phase == "ark":
        return ark[rnd]
    if phase == "sb":
        if rnd == 0: raise SystemExit("SB not defined for round 0")
        return sb[rnd]
    if phase == "sr":
        if rnd == 0: raise SystemExit("SR not defined for round 0")
        return sr[rnd]
    if phase == "mc":
        if rnd in (0,10): raise SystemExit("MC not defined for round 0 or 10")
        return mc[rnd]
    raise SystemExit(f"Unknown phase '{phase}'")

# ================= main =================
def main():
    ap = argparse.ArgumentParser(
        description="Verify CUDA bitslice outputs: final AES or per-round debug dumps."
    )
    ap.add_argument("--meta", required=True, help="inputs/.../meta.json")
    ap.add_argument("--file", required=True,
                    help="File to verify: either final output (groups*128 u32) or stage dump(s).")
    ap.add_argument("--keyhex", default="2b7e151628aed2a6abf7158809cf4f3c", help="AES-128 key hex")
    ap.add_argument("--phase", default="final",
                    choices=["final","ct","ark","sb","sr","mc"],
                    help="What the file contains. 'final'/'ct' compare to AES ciphertext.")
    ap.add_argument("--round", type=int, default=-1,
                    help="Round index (0..10) for ark/sb/sr/mc. Ignored for final/ct.")
    ap.add_argument("--assume_identity", action="store_true",
                    help="For phase=final only: compare CUDA-unpacked blocks to plaintexts.")
    ap.add_argument("--layout", choices=["auto",
                                         "byte_lsb_fwd","byte_lsb_rev","byte_msb_fwd","byte_msb_rev",
                                         "bit_lsb_fwd","bit_lsb_rev","bit_msb_fwd","bit_msb_rev"],
                    default="auto",
                    help="Bitslice layout for unpacking CUDA slices. 'auto' detects from inputs.")
    # Combined-dump support
    ap.add_argument("--combined", action="store_true",
                    help="Interpret --file as a combined dump (groups*dumps_per_group*128 u32).")
    ap.add_argument("--dumps_per_group", type=int, default=11,
                    help="#stages per group in a combined dump (default 11: ARK@0..10).")
    ap.add_argument("--stage_index", type=int, default=0,
                    help="When --combined, select which stage index [0..dpg-1] to compare.")
    args = ap.parse_args()

    # Load meta & inputs
    meta = json.load(open(args.meta, "r"))
    inputs_dir = os.path.dirname(args.meta)
    pt_path   = os.path.join(inputs_dir, "plaintexts.bin")
    inp_path  = os.path.join(inputs_dir, "slices_u32_le.bin")  # used for layout auto-detect

    # Plaintexts
    pts = open(pt_path, "rb").read()
    pt_blocks = [pts[i:i+16] for i in range(0, len(pts), 16)]

    key = bytes.fromhex(args.keyhex)

    # Decide expected blocks
    if args.phase in ("final", "ct"):
        if args.assume_identity:
            exp_blocks = pt_blocks
        else:
            exp_blocks = expected_blocks_for("final", 10, pt_blocks, key)
    else:
        if not (0 <= args.round <= 10):
            raise SystemExit("--round must be 0..10 for ark/sb/sr/mc")
        exp_blocks = expected_blocks_for(args.phase, args.round, pt_blocks, key)

    # Determine layout from input slices (ground truth packing)
    layout: Optional[Layout] = None
    if args.layout == "auto":
        if os.path.exists(inp_path):
            raw = open(inp_path, "rb").read()
            in_u32 = [struct.unpack_from("<I", raw, 4*i)[0] for i in range(len(raw)//4)]
            if len(in_u32) % 128 == 0:
                layout = detect_layout_from_inputs(pt_blocks, in_u32)
        if not layout:
            print("[layout] auto-detect failed; using byte_major/lsb/fwd")
            layout = ("byte_major","lsb","fwd")
    else:
        lut = {
            "byte_lsb_fwd": ("byte_major","lsb","fwd"),
            "byte_lsb_rev": ("byte_major","lsb","rev"),
            "byte_msb_fwd": ("byte_major","msb","fwd"),
            "byte_msb_rev": ("byte_major","msb","rev"),
            "bit_lsb_fwd":  ("bit_major","lsb","fwd"),
            "bit_lsb_rev":  ("bit_major","lsb","rev"),
            "bit_msb_fwd":  ("bit_major","msb","fwd"),
            "bit_msb_rev":  ("bit_major","msb","rev"),
        }
        layout = lut[args.layout]
        print(f"[layout] using explicit: {layout}")

    order, bitorder, lane = layout

    # Load CUDA slices: final or stage(s)
    # If --combined: extract stage_index for each group -> a groups*128 view
    raw_words = read_slices_u32_le(args.file)
    if args.combined:
        dpg = args.dumps_per_group
        if len(raw_words) % (128*dpg) != 0:
            raise SystemExit(f"{args.file}: size {len(raw_words)} not divisible by 128*dpg (dpg={dpg})")
        groups = len(raw_words) // (128*dpg)
        if not (0 <= args.stage_index < dpg):
            raise SystemExit(f"--stage_index must be in [0,{dpg-1}]")
        # carve out the chosen stage
        stage_words: List[int] = []
        stride = 128 * dpg
        base_off = args.stage_index * 128
        for g in range(groups):
            gbase = g * stride + base_off
            stage_words.extend(raw_words[gbase:gbase+128])
        out_slices = stage_words
    else:
        if len(raw_words) % 128 != 0:
            raise SystemExit(f"{args.file}: length {len(raw_words)} not multiple of 128")
        out_slices = raw_words
        groups = len(out_slices) // 128

    # Unpack CUDA output blocks with detected layout
    cuda_blocks: List[bytes] = []

    for g in range(groups):
        gs = out_slices[128*g:128*(g+1)]
        cuda_blocks.extend(unpack_group(gs, order, bitorder, lane))

    # Align lengths
    if len(cuda_blocks) != len(exp_blocks):
        n = min(len(cuda_blocks), len(exp_blocks))
        print(f"[warn] CUDA blocks={len(cuda_blocks)} != expected={len(exp_blocks)}; truncating to {n}.")
        cuda_blocks = cuda_blocks[:n]
        exp_blocks  = exp_blocks[:n]

    # Compare
    title = f"{args.phase.upper()}" if args.phase!="final" else ("IDENTITY" if args.assume_identity else "FINAL")
    print(f"== VERIFY: {title} ==")
    if args.phase not in ("final","ct"):
        print(f"(round = {args.round})")
    diff = compare_blocks(cuda_blocks, exp_blocks)
    print(f"  CUDA: first Group Example hex: {cuda_blocks[0].hex()}")
    print(f"  REF: first Group Example hex: {exp_blocks[0].hex()}")

    print("Match? ", "YES" if diff.ok else "NO")

    if not diff.ok:
        print(f"  first mismatch at block {diff.block}, byte {diff.byte}")
        print(f"  CUDA block hex: {diff.got_hex}")
        print(f"  EXP  block hex: {diff.exp_hex}")

    # Write side-by-side binaries for quick inspection
    out_dir = os.path.dirname(args.file) or "."
    cuda_blob = b"".join(cuda_blocks)
    exp_blob  = b"".join(exp_blocks)
    with open(os.path.join(out_dir, "from_cuda.bin"), "wb") as f:
        f.write(cuda_blob)
    with open(os.path.join(out_dir, "expected.bin"), "wb") as f:
        f.write(exp_blob)

    # SHA
    print("SHA256 CUDA : ", sha256_hex(cuda_blob))
    print("SHA256 EXP  : ", sha256_hex(exp_blob))
    print("MATCH?      : ", "YES" if cuda_blob == exp_blob else "NO")

if __name__ == "__main__":
    main()
