#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, datetime as dt
from bitslice32 import prepare_inputs

def parse_xyz(s: str) -> tuple[int,int,int]:
    parts = s.lower().split("x")
    if len(parts) not in (1,3):
        raise argparse.ArgumentTypeError("Use Nx or NxMxK (e.g., 256 or 128x2x1)")
    if len(parts) == 1:
        n = int(parts[0])
        return (n,1,1)
    return (int(parts[0]), int(parts[1]), int(parts[2]))

def main():
    ap = argparse.ArgumentParser(description="Generate 32-way bitsliced AES inputs based on grid/block.")
    ap.add_argument("--grid", type=parse_xyz, default="1x1x1",
                    help="grid dim as G or GxHxK (default 1x1x1)")
    ap.add_argument("--block", type=parse_xyz, default="256x1x1",
                    help="block dim as B or BxByBz (default 256x1x1)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for deterministic plaintexts (optional)")
    ap.add_argument("--outdir", default=None,
                    help="output directory (default: inputs/run_YYYYmmdd_HHMMSS)")
    args = ap.parse_args()

    outdir = args.outdir or os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "inputs", "run_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ))
    meta = prepare_inputs(outdir, args.grid, args.block, seed=args.seed)
    print("Wrote inputs to:", outdir)
    print("groups =", meta["groups"], "total_blocks =", meta["total_blocks"])

if __name__ == "__main__":
    main()
