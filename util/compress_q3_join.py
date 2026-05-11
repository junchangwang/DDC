#!/usr/bin/env python3
"""
Compress Q3 join_result bitmap into ComBit/WAH/CRoaring/EWAH formats
using the existing compress_tpch_q1 tool's infrastructure.

Since compress_tpch_q1 expects returnflag/linestatus/shipdate structure,
we create a minimal wrapper that treats join_result as a single "column".

Usage: python3 compress_q3_join.py <input_raw_bm> <output_base> <num_rows>
"""
import sys, os, subprocess, struct, shutil, time
import numpy as np


def read_raw_bm(path, num_rows):
    """Read LSB-first packed bitmap file into bool array."""
    data = np.fromfile(path, dtype=np.uint8)
    bits = np.unpackbits(data, bitorder='little')[:num_rows]
    return bits


def write_combit_bm(bits, path):
    """Write as ComBit serialized format using the combit library via a temp tool."""
    packed = np.packbits(bits.astype(np.uint8), bitorder='little')
    packed.tofile(path)


def main():
    if len(sys.argv) < 4:
        print("Usage: compress_q3_join.py <raw_dir> <output_base> <num_rows>")
        sys.exit(1)

    raw_dir = sys.argv[1]
    output_base = sys.argv[2]
    num_rows = int(sys.argv[3])

    raw_bm = os.path.join(raw_dir, "join_result", "0.bm")
    if not os.path.exists(raw_bm):
        print(f"Error: {raw_bm} not found")
        sys.exit(1)

    # Create temp structure that compress_tpch_q1 expects
    # Use a fake column name "join_result" treated as if it were "returnflag" with value 0
    tmpdir = f"/tmp/q3_compress_{os.getpid()}"
    os.makedirs(f"{tmpdir}/returnflag", exist_ok=True)
    os.makedirs(f"{tmpdir}/linestatus", exist_ok=True)
    os.makedirs(f"{tmpdir}/shipdate", exist_ok=True)

    # Copy join_result as returnflag/0.bm
    shutil.copy2(raw_bm, f"{tmpdir}/returnflag/0.bm")

    # Create dummy linestatus/0.bm (all zeros)
    dummy = np.zeros(num_rows, dtype=np.uint8)
    np.packbits(dummy, bitorder='little').tofile(f"{tmpdir}/linestatus/0.bm")

    # Run compress_tpch_q1 (it will compress returnflag/0.bm for us)
    build_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    compress_tool = os.path.join(build_dir, "build", "compress_tpch_q1")

    print(f"[compress] Running compress_tpch_q1 on join_result bitmap...")
    result = subprocess.run(
        [compress_tool, tmpdir, output_base, str(num_rows)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[error] {result.stderr}")
        sys.exit(1)

    # Move the compressed returnflag/0.bm to join_result/0.bm in each output dir
    for fmt in ["combit", "wah", "croaring", "ewah"]:
        out_dir = f"{output_base}_{fmt}"
        src = os.path.join(out_dir, "returnflag", "0.bm")
        dst_dir = os.path.join(out_dir, "join_result")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, "0.bm")
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"  {fmt}: join_result/0.bm = {os.path.getsize(dst)} bytes")
        # Clean up fake dirs
        for d in ["returnflag", "linestatus", "shipdate"]:
            dd = os.path.join(out_dir, d)
            if os.path.isdir(dd) and not os.path.islink(dd):
                shutil.rmtree(dd)

    shutil.rmtree(tmpdir)
    print("[done]")


if __name__ == '__main__':
    main()
