#!/bin/bash

# Script: gen_bitmap.sh
# Description: Generate bitmap files from an existing dataset
#
# Options:
#   -n : Total number of rows (default: 100000000)
#   -c : Cardinality (default: 100)
#   -d : Output base directory (default: .)
#   -z : Zip length - bitmaps per file (default: 1, generates .bm files)
#        When z=1: generates individual .bm files
#        When z>1: generates merged .bmz files
#
# Example usage:
#   ./gen_bitmap.sh -n 1000000 -c 100
#   ./gen_bitmap.sh -n 100000000 -c 10000 -z 500

# Default parameters
n=100000000
c=100
base_dir="."
z=1

# Command parameter parsing
while getopts "n:c:d:z:" opt; do
	case $opt in
		n) n=$OPTARG ;;
		c) c=$OPTARG ;;
		d) base_dir=$OPTARG ;;
		z) z=$OPTARG ;;
		\?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
	esac
done

# Parameter validation
if [ "$n" -le 0 ]; then
	echo "Error: Number of rows must be positive"
	exit 1
fi

if [ "$c" -le 0 ]; then
	echo "Error: Cardinality must be positive"
	exit 1
fi

if [ "$z" -le 0 ]; then
	echo "Error: Zip length must be positive"
	exit 1
fi

# Derived paths
dataset_file="${base_dir}/dataset_${n}_${c}"
output_dir="${base_dir}/bitmap_n${n}_c${c}_z${z}"

# Check dataset exists
if [ ! -f "$dataset_file" ]; then
	echo "Error: Dataset file not found: $dataset_file"
	echo "Please run gen_dataset.sh first."
	exit 1
fi

# Check if bitmap directory already exists
if [ -d "$output_dir" ]; then
	echo "Bitmap directory already exists: $output_dir"
	echo "Skipping generation."
	exit 0
fi

# Derived parameters
packed_bytes=$(( (n + 7) / 8 ))
padded_bits=$(( packed_bytes * 8 ))
num_files=$(( (c + z - 1) / z ))

# Create output directory
mkdir -p "$output_dir"

# Step 1: Generate bitmap files
echo "Step 1: Generating bitmap files..."
echo "Dataset:   $dataset_file"
echo "Output:    $output_dir"
echo "Zip length: $z"
echo "Total bitmaps: $c"
echo "Number of files: $num_files"
echo "Packed bytes per bitmap: $packed_bytes"
echo "Padded bits per bitmap: $padded_bits"

if [ "$z" -eq 1 ]; then
	# Generate individual .bm files
	echo "Mode: individual .bm files"
	python3 -c "
import numpy as np

n = $n
c = $c
output_dir = '$output_dir'
dataset_file = '$dataset_file'

with open(dataset_file, 'rb') as f:
    data = np.fromfile(f, dtype=np.int32)

for val in range(1, c + 1):
    bitmap = np.packbits(data == val, bitorder='little')
    bm_file = f'{output_dir}/{val}.bm'
    with open(bm_file, 'wb') as f:
        f.write(bitmap.tobytes())

    if val % 1000 == 0 or val == c:
        print(f'Generated {val}/{c} .bm files')

print(f'All {c} .bm files generated successfully')
"
else
	# Generate merged .bmz files
	echo "Mode: merged .bmz files"
	python3 -c "
import numpy as np

n = $n
c = $c
z = $z
output_dir = '$output_dir'
dataset_file = '$dataset_file'
num_files = $num_files

with open(dataset_file, 'rb') as f:
    data = np.fromfile(f, dtype=np.int32)

for i in range(num_files):
    start_val = i * z + 1
    end_val = min((i + 1) * z, c)
    bmz_file = f'{output_dir}/{i}.bmz'

    with open(bmz_file, 'wb') as fout:
        for val in range(start_val, end_val + 1):
            bitmap = np.packbits(data == val, bitorder='little')
            fout.write(bitmap.tobytes())

    print(f'Generated bitmaps {start_val}-{end_val} into {i}.bmz')

print(f'All {c} bitmaps merged into {num_files} .bmz files successfully')
"
fi

if [ $? -ne 0 ]; then
	echo "Error: Bitmap generation failed"
	exit 1
fi

# Step 2: Generate index.txt
echo "Step 2: Generating index..."
index_file="${output_dir}/index.txt"
{
	echo "# Bitmap Index File"
	echo "# Generated: $(date)"
	echo ""
	echo "rows = $n"
	echo "cardinality = $c"
	echo "zip_length = $z"
	echo "num_files = $num_files"
	echo "packed_bytes_per_bitmap = $packed_bytes"
	echo "padded_bits_per_bitmap = $padded_bits"
	echo ""
	if [ "$z" -eq 1 ]; then
		echo "# BM file mapping (each file contains 1 bitmap):"
		for ((i=1; i<=c; i++)); do
			echo "${i}.bm: value=${i} bit_start=0 bit_end=$(( n - 1 )) BIT_end=$(( padded_bits - 1 )) BYTE_end=$(( packed_bytes - 1 ))"
		done
	else
		echo "# BMZ file mapping:"
		for ((i=0; i<num_files; i++)); do
			start_val=$(( i * z + 1 ))
			end_val=$(( (i + 1) * z ))
			if [ $end_val -gt $c ]; then
				end_val=$c
			fi
			echo ""
			echo "${i}.bmz: values ${start_val}-${end_val}"
			for ((v=start_val; v<=end_val; v++)); do
				local_idx=$(( v - start_val ))
				bit_start=$(( local_idx * padded_bits ))
				bit_end=$(( bit_start + n - 1 ))
				BIT_end=$(( bit_start + padded_bits - 1 ))
				BYTE_end=$(( local_idx * packed_bytes + packed_bytes - 1 ))
				echo "  value=${v} bit_start=${bit_start} bit_end=${bit_end} BIT_end=${BIT_end} BYTE_end=${BYTE_end}"
			done
		done
	fi
} > "$index_file"

echo "Index file created: $index_file"

# Complete
echo ""
echo "========================================"
echo "Bitmap generation completed successfully!"
echo "========================================"
echo "Output directory: $output_dir"
echo "Dataset file: $dataset_file"
echo "Index file: $index_file"
echo ""
echo "To use the bitmaps, refer to the index file for directory mapping."