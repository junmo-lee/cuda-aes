# bitslice32.py
from __future__ import annotations
from typing import List, Tuple
import os, random, json

def pack_32x128_bitslices(blocks: List[bytes]) -> list[int]:
    if len(blocks) != 32:
        raise ValueError("Expected exactly 32 blocks")
    for b in blocks:
        if len(b) != 16:
            raise ValueError("Each block must be 16 bytes")
    slices = [0] * 128
    for j, blk in enumerate(blocks):
        for byte in range(16):
            v = blk[byte]
            for bit in range(8):  # LSB-first
                s = byte * 8 + bit
                slices[s] |= ((v >> bit) & 1) << j
    return slices

def unpack_32x128_bitslices(slices: List[int]) -> List[bytes]:
    if len(slices) != 128:
        raise ValueError("Expected exactly 128 slices")
    out = [bytearray(16) for _ in range(32)]
    for j in range(32):
        for byte in range(16):
            val = 0
            for bit in range(8):
                s = byte * 8 + bit
                lane_bit = (slices[s] >> j) & 1
                val |= lane_bit << bit
            out[j][byte] = val
    return [bytes(b) for b in out]

def _rng_bytes(n: int, seed: int | None) -> bytes:
    if seed is None:
        return os.urandom(n)
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(n))

def gen_blocks(num_blocks: int, seed: int | None) -> List[bytes]:
    return [_rng_bytes(16, None if seed is None else seed + i) for i in range(num_blocks)]

def group_into_32(blocks: List[bytes]) -> List[List[bytes]]:
    if len(blocks) % 32 != 0:
        raise ValueError("Total blocks must be a multiple of 32")
    return [blocks[i:i+32] for i in range(0, len(blocks), 32)]

def make_batch_slices(groups: List[List[bytes]]) -> List[int]:
    out = []
    for g in groups:
        out.extend(pack_32x128_bitslices(g))
    return out

def write_plaintexts_bin(path: str, blocks: List[bytes]) -> None:
    with open(path, "wb") as f:
        for b in blocks:
            f.write(b)

def write_plaintexts_hex(path: str, blocks: List[bytes]) -> None:
    with open(path, "w") as f:
        for b in blocks:
            f.write(b.hex() + "\n")

def write_slices_u32_le(path: str, slices: List[int]) -> None:
    with open(path, "wb") as f:
        for x in slices:
            f.write(int(x & 0xFFFFFFFF).to_bytes(4, "little"))

def read_slices_u32_le(path: str) -> List[int]:
    out = []
    with open(path, "rb") as f:
        while True:
            b = f.read(4)
            if not b: break
            out.append(int.from_bytes(b, "little"))
    return out

def write_meta_json(path: str, meta: dict) -> None:
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

def prepare_inputs(out_dir: str,
                   grid: tuple[int,int,int],
                   block: tuple[int,int,int],
                   seed: int | None = None) -> dict:
    gx, gy, gz = grid
    bx, by, bz = block
    groups = gx * gy * gz * bx * by * bz
    total_blocks = groups * 32
    blocks = gen_blocks(total_blocks, seed)
    batches = group_into_32(blocks)
    slices = make_batch_slices(batches)

    os.makedirs(out_dir, exist_ok=True)
    write_plaintexts_bin(os.path.join(out_dir, "plaintexts.bin"), blocks)
    # write_plaintexts_hex(os.path.join(out_dir, "plaintexts.hex"), blocks)
    write_slices_u32_le(os.path.join(out_dir, "slices_u32_le.bin"), slices)

    meta = {
        "grid": grid, "block": block,
        "groups": groups, "total_blocks": total_blocks,
        "bytes_per_block": 16,
        "lanes_per_group": 32,
        "slices_per_group": 128,
        "slice_bit_order": "LSB-first (bit s=8*byte+bit)",
        "ordering": "groups in thread-index order; within each group: slices[0..127]",
        "files": {
            "plaintexts.bin": "concatenated 16-byte blocks (total_blocks * 16 bytes)",
            # "plaintexts.hex": "hex-encoded, one block per line",
            "slices_u32_le.bin": "groups*128 little-endian uint32 slices"
        }
    }
    write_meta_json(os.path.join(out_dir, "meta.json"), meta)
    return meta
