#!/bin/bash
# Generate small test bitmap files (.bm) for correctness testing.
# Format: raw packed bits (little-endian bit order), same as gen_bitmap.sh.
#
# Usage:  ./gen_test_bitmaps.sh [-d output_dir]
# Default output: ./test_bitmaps/

OUTPUT_DIR="./test_bitmaps"

while getopts "d:" opt; do
    case $opt in
        d) OUTPUT_DIR=$OPTARG ;;
        \?) echo "Usage: $0 [-d output_dir]" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

python3 << 'PYEOF'
import os, sys, struct, random

output_dir = os.environ.get('OUTPUT_DIR', './test_bitmaps')
os.makedirs(output_dir, exist_ok=True)

n = 1000  # rows per bitmap

def packbits_le(bits):
    """Pack a list of 0/1 into bytes, little-endian bit order (same as numpy packbits bitorder='little')."""
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits) and bits[i + j]:
                byte |= (1 << j)
        result.append(byte)
    return bytes(result)

def write_bm(path, bits):
    with open(path, 'wb') as f:
        f.write(packbits_le(bits))

# ---- Pattern bitmaps ----

# 1. All zeros
write_bm(f'{output_dir}/all_zeros.bm', [0] * n)

# 2. All ones
write_bm(f'{output_dir}/all_ones.bm', [1] * n)

# 3. Sparse (~1% density)
random.seed(42)
sparse = [1 if random.random() < 0.01 else 0 for _ in range(n)]
write_bm(f'{output_dir}/sparse_1pct.bm', sparse)

# 4. Dense (~90% density)
random.seed(43)
dense = [1 if random.random() < 0.90 else 0 for _ in range(n)]
write_bm(f'{output_dir}/dense_90pct.bm', dense)

# 5. Alternating 010101...
alternating = [i % 2 for i in range(n)]
write_bm(f'{output_dir}/alternating.bm', alternating)

# 6. Single bit set (position 500)
single = [0] * n
single[500] = 1
write_bm(f'{output_dir}/single_bit_500.bm', single)

# ---- Simulated column (like gen_bitmap.sh) ----
random.seed(99)
data = [random.randint(1, 5) for _ in range(n)]

os.makedirs(f'{output_dir}/column', exist_ok=True)
for val in range(1, 6):
    bitmap = [1 if d == val else 0 for d in data]
    write_bm(f'{output_dir}/column/{val}.bm', bitmap)

# Write metadata
with open(f'{output_dir}/metadata.txt', 'w') as f:
    f.write(f'num_rows={n}\n')
    f.write(f'cardinality=5\n')
    f.write(f'all_zeros_popcount=0\n')
    f.write(f'all_ones_popcount={n}\n')
    f.write(f'single_bit_500_popcount=1\n')
    f.write(f'alternating_popcount={n // 2}\n')
    f.write(f'sparse_1pct_popcount={sum(sparse)}\n')
    f.write(f'dense_90pct_popcount={sum(dense)}\n')
    for val in range(1, 6):
        cnt = sum(1 for d in data if d == val)
        f.write(f'column_{val}_popcount={cnt}\n')

print(f'Generated test bitmaps in {output_dir}/')
print(f'  Pattern files: all_zeros, all_ones, sparse_1pct, dense_90pct, alternating, single_bit_500')
print(f'  Column files:  column/1..5.bm')
PYEOF
