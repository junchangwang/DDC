#!/bin/bash

# Script: gen_bitmap.sh
# Description: Script for bitmap generation from dataset_* files with configurable parameters
#
# Options:
#   -n : Total number of rows (default: 100000000)
#   -c : Cardinality (default: 100)
#
# Example usage:
#   ./gen_bitmap.sh

n=100000000
c=10000000

# 1) generate a dataset with 100M rows and 100 cardinality

# 2) generate raw bitmap from dataset

# bitmap/1.bm bitmap/2.bm 10000000.bm

# I have 1000000 files, how to manage (store in disk) them efficiently?

# bmz (core idea: 100000 bm files -> 1 bmz file)
