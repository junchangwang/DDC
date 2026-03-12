#!/bin/bash

# GHz
if grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null; then
    CPU_GHZ=2.00 # default for WSL
elif [[ -f /proc/cpuinfo ]]; then
    CPU_GHZ=$(grep -m 1 'cpu MHz' /proc/cpuinfo | awk '{print $4 / 1000}')
    if [[ -z "$CPU_GHZ" ]]; then
        CPU_GHZ=$(grep -m 1 'model name' /proc/cpuinfo | awk -F '@' '{print $2}' | tr -d 'GHz ')
    fi
    if [[ -z "$CPU_GHZ" ]]; then
        CPU_GHZ=2.00
    fi
else
    CPU_GHZ=2.00 # default
fi

echo "$CPU_GHZ"
