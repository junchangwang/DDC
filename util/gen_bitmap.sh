#!/bin/bash

# Script: gen_bitmap.sh
# Description: Script for bitmap generation with efficient file management
#
# Options:
#   -n : Total number of rows (default: 100000000)
#   -c : Cardinality (default: 100)
#   -d : Output directory (default: ./bitmaps)
#   -s : Files per subdirectory (default: 1000)
#
# Example usage:
#   ./gen_bitmap.sh -n 10000000 -c 100
#   ./gen_bitmap.sh -n 100000000 -c 10000 -d ./my_bitmaps -s 500

# Default parameters
n=100000000
c=100
output_dir="./bitmaps"
files_per_dir=1000

#Command parameter parsing
while getopts "n:c:d:s:" opt; do
	case $opt in
		n)
			n=$OPTARG
			;;
		c)
			c=$OPTARG
			;;
		d)
			output_dir=$OPTARG
			;;
		s)
			files_per_dir=$OPTARG
			;;
		\?)
			echo "Invalid option: -$OPTARG" >&2
			exit 1
			;;
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

if [ "$files_per_dir" -le 0 ]; then
	echo "Error: Files per directory must be positive"
	exit 1
fi

# Output root directory
mkdir -p "$output_dir"

# Generate dataset filename
dataset_file="dataset_${n}_${c}"

# Step 1: Generate dataset (if not exists)
echo "Step 1: Generating dataset with $n rows and $c cardinality..."
if [ ! -f "$dataset_file" ]; then
	python3 -c "
import numpy as np
import struct

n = $n
c = $c

# Generate random data (uniform distribution)
data = np.random.randint(1, c + 1, size=n, dtype=np.int32)

# Save as binary file
with open('$dataset_file', 'wb') as f:
    f.write(data.tobytes())

print(f'Dataset generated: {n} rows, {c} cardinality')
"
	if [ $? -eq 0 ]; then
		echo "Dataset generated successfully: $dataset_file"
	else
		echo "Error: Failed to generate dataset"
		exit 1
	fi
else
	echo "Dataset already exists: $dataset_file"
fi

# Step 2: Generate bitmap files (hierarchical management)
echo "Step 2: Generating bitmap files with hierarchical management..."
echo "Output directory: $output_dir"
echo "Files per subdirectory: $files_per_dir"
echo "Total bitmap files to generate: $c"

# Calculate number of subdirectories needed
#Divisibility by up
num_dirs=$(( (c + files_per_dir - 1) / files_per_dir ))
echo "Number of subdirectories: $num_dirs"

# Create subdirectories
for ((i=0; i<num_dirs; i++)); do
	subdir="${output_dir}/part_${i}"
	mkdir -p "$subdir"
	echo "Created subdirectory: $subdir"
done

# Generate bitmap files
echo "Generating bitmap files..."

# use compiled C++ program
#name: nicolas
if [ -f "../build/nicolas" ]; then
	# Use compiled program
	for ((i=0; i<num_dirs; i++)); do
		subdir="${output_dir}/part_${i}"
		start_val=$(( i * files_per_dir + 1 ))
		end_val=$(( (i + 1) * files_per_dir ))
		if [ $end_val -gt $c ]; then
			end_val=$c
		fi
		../build/nicolas \
			--cardinality $c \
			--index-path "$subdir" \
			--number-of-rows $n \
			--data-path "$dataset_file" \
			--files-per-dir $files_per_dir \
			--start-val $start_val \
			--end-val $end_val
	done
	
	if [ $? -eq 0 ]; then
		echo "Bitmap generation completed successfully"
	else
		echo "Error: Bitmap generation failed"
		exit 1
	fi
else
	# If no compiled program, use Python to generate simple bitmaps
	echo "Warning: C++ program not found, using Python for bitmap generation"
	python3 -c "
import numpy as np
import struct
import os

n = $n
c = $c
output_dir = '$output_dir'
files_per_dir = $files_per_dir

# Read dataset
with open('$dataset_file', 'rb') as f:
    data = np.fromfile(f, dtype=np.int32)

# Generate bitmap for each value
for val in range(c):
    # Calculate subdirectory and filename
    dir_idx = val // files_per_dir
    subdir = f'{output_dir}/part_{dir_idx}'
    os.makedirs(subdir, exist_ok=True)
    
    # Create bitmap (set positions with value val+1 to 1)
    bitmap = np.packbits(data == val + 1, bitorder='little')
    
    # Save bitmap
    bitmap_file = f'{subdir}/{val + 1}.bm'
    with open(bitmap_file, 'wb') as f:
        f.write(bitmap.tobytes())
    
    if (val + 1) % 1000 == 0:
        print(f'Generated {val + 1}/{c} bitmaps')

print(f'All {c} bitmaps generated successfully')
"
	
	if [ $? -eq 0 ]; then
		echo "Bitmap generation completed successfully"
	else
		echo "Error: Bitmap generation failed"
		exit 1
	fi
fi

# Step 3: Generate directory index file (for fast lookup)
echo "Step 3: Generating directory index..."
index_file="${output_dir}/index.txt"
{
	echo "# Bitmap Index File"
	echo "# Generated: $(date)"
	echo "# Total rows: $n"
	echo "# Cardinality: $c"
	echo "# Files per directory: $files_per_dir"
	echo "# Number of directories: $num_dirs"
	echo ""
	echo "# Directory mapping:"
	for ((i=0; i<num_dirs; i++)); do
		start_val=$(( i * files_per_dir + 1 ))
		end_val=$(( (i + 1) * files_per_dir ))
		if [ $end_val -ge $c ]; then
			end_val=$c
		fi
		echo "part_${i}: values ${start_val}-${end_val}"
	done
} > "$index_file"

echo "Index file created: $index_file"

# Step 4: Generate statistics information
echo "Step 4: Generating statistics..."
stats_file="${output_dir}/stats.txt"
{
	echo "Bitmap Generation Statistics"
	echo "========================"
	echo "Generated: $(date)"
	echo "Total rows: $n"
	echo "Cardinality: $c"
	echo "Output directory: $output_dir"
	echo "Files per directory: $files_per_dir"
	echo "Number of directories: $num_dirs"
	echo "Total bitmap files: $c"
	echo ""
	echo "Storage information:"
	du -sh "$output_dir"
	echo ""
	echo "File count per directory:"
	for ((i=0; i<num_dirs; i++)); do
		subdir="${output_dir}/part_${i}"
		if [ -d "$subdir" ]; then
			count=$(find "$subdir" -name "*.bm" | wc -l)
			echo "  part_${i}: $count files"
		fi
	done
} > "$stats_file"

echo "Statistics file created: $stats_file"

# complete
echo ""
echo "========================================"
echo "Bitmap generation completed successfully!"
echo "========================================"
echo "Output directory: $output_dir"
echo "Dataset file: $dataset_file"
echo "Index file: $index_file"
echo "Statistics file: $stats_file"
echo ""
echo "To use the bitmaps, refer to the index file for directory mapping."