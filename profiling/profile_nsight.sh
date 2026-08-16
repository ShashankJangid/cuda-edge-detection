#!/usr/bin/env bash
# Nsight Profiling Automation Script for CUDA Edge Detection
set -euo pipefail

IMAGE=${1:-"../data/input/sample.png"}
OUT_DIR="./profiling_results"
mkdir -p "$OUT_DIR"

echo "=== 1. NVIDIA Nsight Systems Timeline Profiling ==="
if command -v nsys &> /dev/null; then
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --stats=true \
        --output="$OUT_DIR/sobel_timeline" \
        --force-overwrite=true \
        ../build/edge_detect sobel "$IMAGE" /dev/null
    echo "✓ Nsight Systems report saved: $OUT_DIR/sobel_timeline.nsys-rep"
else
    echo "⚠ Nsight Systems (nsys) not found on PATH."
fi

echo "=== 2. NVIDIA Nsight Compute Kernel & Roofline Analysis ==="
if command -v ncu &> /dev/null; then
    ncu \
        --set full \
        --section SpeedOfLight \
        --section MemoryWorkloadAnalysis \
        --section Occupancy \
        --export="$OUT_DIR/roofline_report" \
        --force-overwrite \
        ../build/edge_detect sobel "$IMAGE" /dev/null
    echo "✓ Nsight Compute Roofline report saved: $OUT_DIR/roofline_report.ncu-rep"
else
    echo "⚠ Nsight Compute (ncu) not found on PATH."
fi
