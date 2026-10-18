#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Build and run the DIRECT ("kernel launch") check of MulAscend - without registering the op in CANN.
#
#   bash test/build_direct_launch.sh                    # all sizes
#   bash test/build_direct_launch.sh --bench            # + kernel timing
#   bash test/build_direct_launch.sh --size 100000      # a single size
#   bash test/build_direct_launch.sh --kernel <path.o>  # an explicit device binary
#
# The device .o produced by `bash build.sh` (build_out/op_kernel/ascendc_kernels/binary/ascend910b/
# mul_ascend/MulAscend_<hash>.o) is loaded with aclrtBinaryLoadFromFile and launched with aclrtLaunchKernel.
# Nothing of the custom-op framework takes part: no aclnn wrapper, no TilingFunc call, no
# custom_opp_*.run install and no ASCEND_CUSTOM_OPP_PATH.
#
# CPU-only machines keep working: the script then exits with code 3 and points to the CPU debug model.
# ----------------------------------------------------------------------------------------------------------
set -e

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)

KERNEL=""
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --kernel)  KERNEL="$2"; shift ;;
        -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)         ARGS+=("$1") ;;
    esac
    shift
done

# --- locate the toolkit ---
CANN="${ASCEND_HOME_PATH:-}"
if [ -z "$CANN" ] || [ ! -d "$CANN/lib64" ]; then
    for c in /home/user/Ascend/ascend-toolkit/cann-9.1.0 \
             /usr/local/Ascend/ascend-toolkit/latest \
             /usr/local/Ascend/latest; do
        if [ -d "$c/lib64" ]; then
            CANN="$c"
            break
        fi
    done
fi
if [ -z "$CANN" ] || [ ! -d "$CANN/lib64" ]; then
    echo "ERROR: the CANN toolkit was not found. Set ASCEND_HOME_PATH." >&2
    exit 2
fi
echo "toolkit    : $CANN"

# --- the kernel binary built by build.sh ---
if [ -z "$KERNEL" ]; then
    KERNEL=$(ls -1 "$ROOT"/build_out/op_kernel/ascendc_kernels/binary/ascend910b/*/*.o 2>/dev/null | head -1)
fi
if [ -z "$KERNEL" ] || [ ! -f "$KERNEL" ]; then
    echo "ERROR: no kernel .o found - run 'bash build.sh' first (or pass --kernel <path.o>)." >&2
    exit 2
fi
echo "kernel .o  : $KERNEL"

# The driver knows the kernel by the name of its .ascend.meta.<name> section, which is also the ELF symbol
# of the entry - take it straight from the binary instead of hard-coding the build hash.
KERNEL_NAME=$(readelf -sW "$KERNEL" | awk '$4 == "FUNC" && $5 == "GLOBAL" { print $8; exit }')
echo "kernel name: ${KERNEL_NAME:-<by entry>}"

# The host passes the five GM_ADDR arguments as one flat 40-byte buffer; the binary states its own
# argument size in the __CCE_KernelArgSize note, so a signature change shows up here immediately.
# readelf prints the four bytes of that note in memory order (little-endian), hence the reversal.
ARG_HEX=$(readelf -x __CCE_KernelArgSize "$KERNEL" 2>/dev/null | sed -n '3p' | awk '{print $2}')
if [ -n "$ARG_HEX" ]; then
    ARG_BYTES=$((16#${ARG_HEX:6:2}${ARG_HEX:4:2}${ARG_HEX:2:2}${ARG_HEX:0:2}))
    echo "args size  : $ARG_BYTES bytes (x, y, z, workspace, tiling)"
    if [ "$ARG_BYTES" != "40" ]; then
        echo "WARNING: the kernel declares $ARG_BYTES argument bytes, but this host passes 5 pointers (40)." >&2
    fi
fi

# --- the host program ---
echo "building   : $HERE/direct_launch_check"
g++ -std=c++17 -O2 -Wall "$HERE/direct_launch_check.cpp" -o "$HERE/direct_launch_check" \
    -I"$CANN/include" -L"$CANN/lib64" \
    -Wl,-rpath,"$CANN/lib64" -Wl,--allow-shlib-undefined \
    -lascendcl -lpthread

# --- runtime environment ---
export LD_LIBRARY_PATH="$CANN/x86_64-linux/lib64:$CANN/lib64:$CANN/runtime/lib64:$LD_LIBRARY_PATH"
for c in /usr/local/Ascend/driver/lib64 "$HOME/Ascend/driver/lib64" "$HOME/Ascend/driver/lib64/driver" \
         "$CANN/../driver/lib64"; do
    if [ -d "$c" ]; then
        export LD_LIBRARY_PATH="$c:$LD_LIBRARY_PATH"
    fi
done

# The ACL runtime needs the driver libraries (libascend_hal.so, ...).  Without them this host has no NPU,
# and we say so instead of failing with a loader error.
if ldd "$HERE/direct_launch_check" 2>/dev/null | grep -q 'not found'; then
    echo "ERROR: the ACL runtime cannot be loaded (missing driver libraries):"
    ldd "$HERE/direct_launch_check" 2>/dev/null | grep 'not found' | sed 's/^/  /'
    echo "       This host has no Ascend NPU driver, so the direct launch cannot run here."
    echo "       CPU-only checks:  bash test/build_cpu_debug.sh  /  bash test/example_trace/run.sh"
    exit 3
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
echo "exit code  : $rc   (0 = ok, 1 = mismatch, 2 = acl error, 3 = no device)"
exit $rc
