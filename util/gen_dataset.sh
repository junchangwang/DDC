#!/bin/bash

# Script: gen_dataset.sh
# Description: Generate a random integer dataset and record in done.txt
#
# Options:
#   -n : Total number of rows (default: 100000000)
#   -c : Cardinality (default: 100)
#   -d : Output base directory (default: .)
#
# Example usage:
#   ./gen_dataset.sh -n 1000000 -c 100
#   ./gen_dataset.sh -n 100000000 -c 10000 -d /data

# Default parameters
n=100000000
c=100
base_dir="."

# Command parameter parsing
while getopts "n:c:d:" opt; do
	case $opt in
		n) n=$OPTARG ;;
		c) c=$OPTARG ;;
		d) base_dir=$OPTARG ;;
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

# Derived paths
dataset_file="${base_dir}/dataset_${n}_${c}"
done_file="${base_dir}/done.txt"

# Check done.txt: skip if dataset already exists
if [ -f "$done_file" ]; then
	if grep -q "n=${n} c=${c}" "$done_file"; then
		echo "Dataset already exists for n=${n}, c=${c}"
		echo "Dataset file: $dataset_file"
		echo "Skipping generation."
		exit 0
	fi
fi

# Generate dataset
echo "Generating dataset with $n rows and $c cardinality..."
python3 -c "
import numpy as np

n = $n
c = $c
dataset_file = '$dataset_file'

data = np.random.randint(1, c + 1, size=n, dtype=np.int32)

with open(dataset_file, 'wb') as f:
    f.write(data.tobytes())

print(f'Dataset generated: {n} rows, {c} cardinality')
"

if [ $? -eq 0 ]; then
	echo "Dataset generated successfully: $dataset_file"
else
	echo "Error: Failed to generate dataset"
	exit 1
fi

# Update done.txt
echo "n=${n} c=${c} dir=${dataset_file}" >> "$done_file"
echo "done.txt updated: $done_file"

echo ""
echo "========================================"
echo "Dataset generation completed successfully!"
echo "========================================"
echo "Dataset file: $dataset_file"