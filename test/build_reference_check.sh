#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Compile and run the host-side check of the int64 multiplication math (src/reference_check.cpp).
# It validates the 32x32 -> 64 decomposition used by op_kernel/mul_ascend.cpp against the
# wrap-around int64 product on 200000 random inputs.
# ----------------------------------------------------------------------------------------------------------
set -e
HERE=$(cd "$(dirname "$0")" && pwd)

g++ -O2 -std=c++17 "$HERE/../src/reference_check.cpp" -o "$HERE/reference_check"
exec "$HERE/reference_check"
