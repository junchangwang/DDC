#!/bin/bash

# Script: del_bitmap.sh
# Description: Delete bitmap directory and/or dataset, sync done.txt
#
# Mode 1 - Delete a single bitmap directory only (requires -z):
#   ./del_bitmap.sh -n <rows> -c <cardinality> -z <zip_length>
#
# Mode 2 - Delete all bitmap directories + dataset, sync done.txt (no -z):
#   ./del_bitmap.sh -n <rows> -c <cardinality>
#
# Options:
#   -n : Total number of rows (required)
#   -c : Cardinality (required)
#   -z : Zip length (optional; if specified: mode 1; if omitted: mode 2)
#   -d : Base directory (default: .)
#
# Example usage:
#   ./del_bitmap.sh -n 1000 -c 20 -z 5    # mode 1: delete bitmap_n1000_c20_z5 only
#   ./del_bitmap.sh -n 1000 -c 20          # mode 2: delete all bitmaps + dataset + done.txt entry

n=""
c=""
z=""
base_dir="."

while getopts "n:c:z:d:" opt; do
	case $opt in
		n) n=$OPTARG ;;
		c) c=$OPTARG ;;
		z) z=$OPTARG ;;
		d) base_dir=$OPTARG ;;
		\?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
	esac
done

# Validate required parameters
if [ -z "$n" ] || [ -z "$c" ]; then
	echo "Error: -n and -c are required"
	echo "Usage:"
	echo "  Mode 1 (single bitmap dir): $0 -n <rows> -c <cardinality> -z <zip_length>"
	echo "  Mode 2 (all bitmaps + dataset): $0 -n <rows> -c <cardinality>"
	exit 1
fi

done_file="${base_dir}/done.txt"

if [ -n "$z" ]; then
	# ---- Mode 1: Delete a single bitmap directory, no done.txt changes ----
	echo "Mode 1: Deleting single bitmap directory"
	output_dir="${base_dir}/bitmap_n${n}_c${c}_z${z}"

	if [ ! -d "$output_dir" ]; then
		echo "Error: Directory not found: $output_dir"
		exit 1
	fi

	echo "Deleting $output_dir ..."
	rm -rf "$output_dir"
	echo "Deleted: $output_dir"

else
	# ---- Mode 2: Delete all bitmap directories + dataset, update done.txt ----
	echo "Mode 2: Deleting all bitmap directories and dataset"

	# Delete all bitmap dirs for this n/c
	found=0
	while IFS= read -r -d '' dir; do
		echo "Deleting $dir ..."
		rm -rf "$dir"
		echo "Deleted: $dir"
		found=1
	done < <(find "$base_dir" -maxdepth 1 -type d -name "bitmap_n${n}_c${c}_z*" -print0)

	if [ "$found" -eq 0 ]; then
		echo "No bitmap directories found for n=${n} c=${c}"
	fi

	# Delete dataset
	dataset_file="${base_dir}/dataset_${n}_${c}"
	if [ -f "$dataset_file" ]; then
		echo "Deleting dataset: $dataset_file ..."
		rm -f "$dataset_file"
		echo "Deleted: $dataset_file"
	else
		echo "Warning: Dataset file not found: $dataset_file"
	fi

	# Update done.txt
	if [ -f "$done_file" ]; then
		sed -i "/n=${n} c=${c} /d" "$done_file"
		echo "done.txt updated: removed entry n=${n} c=${c}"
	else
		echo "Warning: done.txt not found, nothing to update"
	fi
fi

echo ""
echo "Done."