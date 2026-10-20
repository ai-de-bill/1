#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# MulAscend on-device check - WITHOUT registering the operator in CANN.
#
#   bash run.sh                          # all sizes on real 910B (uses the prebuilt kernel binary)
#   bash run.sh --size 100000            # a single size
#   bash run.sh --bench 20               # + kernel timing (us/op, ns/element)
#   bash run.sh --build-kernel           # rebuild the kernel .o from kernel/ first, then run
#   bash run.sh --check-inputs           # diagnostics: device addresses + host-to-device read-back
#   bash run.sh --kernel <path.o>        # an explicit device binary
#   bash run.sh --sim                    # run on the camodel NPU simulator instead of real hardware
#
# The kernel is started directly, the way the official MindStudio "kernel launch" tool (mskl) does it:
# rtRegisterAllKernel() + rtKernelLaunchWithHandleV2().  No custom_opp_*.run install, no aclnn wrapper,
# no ASCEND_CUSTOM_OPP_PATH, no TilingFunc call at run time.
#
# The kernel entry is mul_ascend(x, y, z, workspace, tiling) - five GM_ADDR, 40 bytes.  `tiling` points at
# a device buffer holding the element count as uint32 at offset 0 (what TilingFunc would have written), so
# the size is chosen here at run time; block-dim is 16 (context->SetBlockDim(16) in op_host).
#
# exit codes: 0 = everything passed, 1 = wrong results, 2 = acl/build error, 3 = no NPU device visible.
# ----------------------------------------------------------------------------------------------------------
set -e

HERE=$(cd "$(dirname "$0")" && pwd)

KERNEL=""
BUILD_KERNEL=0
SIM_MODE=0
SIM_SOC="Ascend910B1"
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --kernel)       KERNEL="$2"; shift ;;
        --build-kernel) BUILD_KERNEL=1 ;;
        --sim)          SIM_MODE=1 ;;
        --sim-soc)      SIM_SOC="$2"; shift ;;
        -h|--help)      sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)              ARGS+=("$1") ;;
    esac
    shift
done

# --- locate the CANN toolkit ---
CANN="${ASCEND_HOME_PATH:-}"
if [ -z "$CANN" ] || [ ! -d "$CANN/lib64" ]; then
    for c in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/latest \
             "$HOME/Ascend/ascend-toolkit/latest" "$HOME/Ascend/ascend-toolkit/cann-9.1.0"; do
        if [ -d "$c/lib64" ]; then
            CANN="$c"
            break
        fi
    done
fi
if [ -z "$CANN" ] || [ ! -d "$CANN/lib64" ]; then
    echo "ERROR: the CANN toolkit was not found - set ASCEND_HOME_PATH." >&2
    exit 2
fi
echo "toolkit    : $CANN"

# --- the kernel binary ---
if [ "$BUILD_KERNEL" = "1" ]; then
    echo "building   : kernel (bash kernel/build.sh, log: kernel_build.log)"
    if ! ( cd "$HERE/kernel" && bash build.sh >"$HERE/kernel_build.log" 2>&1 ); then
        echo "ERROR: the kernel build failed - see $HERE/kernel_build.log" >&2
        tail -20 "$HERE/kernel_build.log" >&2
        exit 2
    fi
fi
if [ -z "$KERNEL" ]; then
    KERNEL=$(ls -1 "$HERE"/kernel/build_out/op_kernel/ascendc_kernels/binary/ascend910b/*/*.o 2>/dev/null | head -1)
fi
if [ -z "$KERNEL" ]; then
    KERNEL=$(ls -1 "$HERE"/prebuilt/*.o 2>/dev/null | head -1)
fi
if [ -z "$KERNEL" ] || [ ! -f "$KERNEL" ]; then
    echo "ERROR: no kernel .o found - pass --kernel <path.o> or use --build-kernel." >&2
    exit 2
fi
echo "kernel .o  : $KERNEL"

# The kernel is not recompiled by this script; warn if the sources are newer than the binary.
SRC_NEWEST=$(ls -t "$HERE"/kernel/op_kernel/*.cpp "$HERE"/kernel/op_kernel/*.h 2>/dev/null | head -1)
if [ -n "$SRC_NEWEST" ] && [ "$SRC_NEWEST" -nt "$KERNEL" ]; then
    echo "WARNING    : $SRC_NEWEST is newer than $KERNEL - rebuild with --build-kernel"
fi

# Kernel name (== the .ascend.meta.<name> section) straight from the ELF symbol.
KERNEL_NAME=$(readelf -sW "$KERNEL" | awk '$4 == "FUNC" && $5 == "GLOBAL" { print $8; exit }')
echo "kernel name: ${KERNEL_NAME:-<by entry>}"

# The binary states its own argument size in the __CCE_KernelArgSize note; readelf prints the four bytes in
# memory order (little-endian), hence the reversal. 5 GM_ADDR = 40 bytes is what this host passes.
ARG_HEX=$(readelf -x __CCE_KernelArgSize "$KERNEL" 2>/dev/null | sed -n '3p' | awk '{print $2}')
if [ -n "$ARG_HEX" ]; then
    ARG_BYTES=$((16#${ARG_HEX:6:2}${ARG_HEX:4:2}${ARG_HEX:2:2}${ARG_HEX:0:2}))
    echo "args size  : $ARG_BYTES bytes (x, y, z, workspace, tiling)"
    if [ "$ARG_BYTES" != "40" ]; then
        echo "WARNING    : the kernel declares $ARG_BYTES argument bytes, but this host passes 5 pointers." >&2
    fi
fi

# --- the host program ---
echo "building   : $HERE/direct_launch_check"
g++ -std=c++17 -O2 -Wall "$HERE/direct_launch_check.cpp" -o "$HERE/direct_launch_check" \
    -I"$CANN/include" -L"$CANN/lib64" \
    -Wl,-rpath,"$CANN/lib64" -Wl,--allow-shlib-undefined \
    -lascendcl -lpthread -ldl

export ASCEND_HOME_PATH="$CANN"

# --- runtime environment ---
if [ "$SIM_MODE" = "1" ]; then
    # camodel NPU simulator: libruntime_camodel.so (emulated runtime) + libnpu_drv.so (emulated driver), which
    # must be visible under the names the ACL runtime uses; LD_PRELOAD makes the emulated runtime win over the
    # real one (this is also how the official mskl tool recognises a simulation run).
    SIM_DIR="$CANN/x86_64-linux/simulator/$SIM_SOC/lib"
    if [ ! -f "$SIM_DIR/libruntime_camodel.so" ]; then
        echo "ERROR: the camodel simulator for '$SIM_SOC' was not found ($SIM_DIR)." >&2
        exit 2
    fi
    mkdir -p "$HERE/.sim_ld" "$HERE/camodel_logs"
    ln -sf "$SIM_DIR/libruntime_camodel.so" "$HERE/.sim_ld/libruntime.so"
    ln -sf "$SIM_DIR/libnpu_drv.so" "$HERE/.sim_ld/libascend_hal.so"
    export LD_PRELOAD="$SIM_DIR/libruntime_camodel.so${LD_PRELOAD:+:$LD_PRELOAD}"
    export LD_LIBRARY_PATH="$HERE/.sim_ld:$SIM_DIR:$CANN/x86_64-linux/lib64:$CANN/lib64:$CANN/runtime/lib64:$LD_LIBRARY_PATH"
    export CAMODEL_LOG_PATH="$HERE/camodel_logs"
    echo "simulator  : $SIM_DIR (camodel $SIM_SOC)"
else
    export LD_LIBRARY_PATH="$CANN/x86_64-linux/lib64:$CANN/lib64:$CANN/runtime/lib64:$LD_LIBRARY_PATH"
    for c in /usr/local/Ascend/driver/lib64 "$HOME/Ascend/driver/lib64" "$HOME/Ascend/driver/lib64/driver" \
             "$CANN/../driver/lib64"; do
        if [ -d "$c" ]; then
            export LD_LIBRARY_PATH="$c:$LD_LIBRARY_PATH"
        fi
    done

    # The ACL runtime needs the driver libraries (libascend_hal.so, ...).  Without them there is no NPU here.
    if ldd "$HERE/direct_launch_check" 2>/dev/null | grep -q 'not found'; then
        echo "ERROR: the ACL runtime cannot be loaded (missing driver libraries):" >&2
        ldd "$HERE/direct_launch_check" 2>/dev/null | grep 'not found' | sed 's/^/  /' >&2
        echo "       This machine has no Ascend NPU driver - install the driver, or use --sim." >&2
        exit 3
    fi
fi

# --- run ---
RUN_ARGS=(--kernel "$KERNEL")
if [ -n "$KERNEL_NAME" ]; then
    RUN_ARGS+=(--name "$KERNEL_NAME")
fi
set +e
"$HERE/direct_launch_check" "${RUN_ARGS[@]}" "${ARGS[@]}"
rc=$?
set -e
echo "exit code  : $rc   (0 = ok, 1 = wrong results, 2 = acl error, 3 = no device)"
exit $rc
