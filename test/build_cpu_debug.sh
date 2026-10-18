#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Build the mul_ascend kernel for the Ascend 910B CPU debug model and run it against a reference.
# Requires the CANN toolkit with tools/cpudebug (libcpudebug.so for Ascend910B1).
# ----------------------------------------------------------------------------------------------------------
set -e

CANN=${ASCEND_HOME_PATH:-/home/user/Ascend/ascend-toolkit/cann-9.1.0}
CPD=$CANN/tools/cpudebug
HERE=$(cd "$(dirname "$0")" && pwd)

g++ -std=c++17 -O2 -w \
    -DASCENDC_CPU_DEBUG=1 -D__CCE_KT_TEST__=1 -D__NPU_ARCH__=2201 -D__CCE_AICORE__=220 \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I"$HERE/../op_kernel" \
    -I"$CPD/include" -I"$CPD/lib/include" \
    -I"$CANN/include" -I"$CANN/include/base" \
    -I"$CANN/compiler/tikcpp/tikcfw" -I"$CANN/compiler/tikcpp/tikcfw/impl" \
    -I"$CANN/compiler/tikcpp/tikcfw/interface" \
    -I"$CANN/compiler/ascendc/include/highlevel_api/lib" \
    -I"$CANN/compiler/ascendc/include/highlevel_api/impl" \
    "$HERE/cpu_debug_check.cpp" -o "$HERE/mul_ascend_cpu_check" \
    -L"$CPD/lib/Ascend910B1" -L"$CPD/lib64" \
    -L"$CANN/x86_64-linux/simulator/dav_2201/lib" \
    -L"$CANN/lib64" -L"$CANN/x86_64-linux/lib64/device/lib64" \
    -Wl,-Bdynamic,--no-as-needed \
    -lcpudebug -lcpudebug_stubreg -lcpudebug_cceprint -lcpudebug_npuchk \
    -lpem_davinci -lc_sec -lmmpa -lunified_dlog -lpthread -Wl,--as-needed

export LD_LIBRARY_PATH="$CPD/lib/Ascend910B1:$CPD/lib64:$CANN/x86_64-linux/simulator/dav_2201/lib:$CANN/lib64:$CANN/x86_64-linux/lib64/device/lib64:$LD_LIBRARY_PATH"
"$HERE/mul_ascend_cpu_check"
rc=$?

# the CPU debug runtime drops these into the working directory
rm -rf cceprint npuchk stub_reg.log
exit $rc
