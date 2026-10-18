#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Build and run the ON-DEVICE (real Ascend 910B) check of the MulAscend custom op.
#
#   bash test/build_hw_check.sh                 # correctness on the default size list
#   bash test/build_hw_check.sh --bench         # correctness + kernel timing
#   bash test/build_hw_check.sh --size 100000   # a single size
#   bash test/build_hw_check.sh --vendor DIR    # explicit .../vendors/customize directory
#
# The custom op package has to be built and installed first:
#   bash build.sh
#   ./build_out/custom_opp_*.run --install --install-path=$HOME/Ascend/custom_opp
#   export ASCEND_CUSTOM_OPP_PATH=$HOME/Ascend/custom_opp/vendors/customize
# The script finds the vendor directory automatically in $ASCEND_CUSTOM_OPP_PATH,
# <CANN>/opp/vendors/customize, $HOME/Ascend/custom_opp/vendors/customize,
# /usr/local/Ascend/opp/vendors/customize, or in the package unpacked by build.sh under build_out/.
#
# CPU-only machines keep working as before: this script then exits with code 3 and points to
#   bash test/build_cpu_debug.sh   /   bash test/example_trace/run.sh
# ----------------------------------------------------------------------------------------------------------
set -e

CANN=${ASCEND_HOME_PATH:-/home/user/Ascend/ascend-toolkit/cann-9.1.0}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)

VENDOR=""
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --vendor)  VENDOR="$2"; shift ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)         ARGS+=("$1") ;;
    esac
    shift
done

# --- locate the installed custom op (needs op_api/include/aclnn_mul_ascend.h) ---
if [ -z "$VENDOR" ]; then
    for c in "$ASCEND_CUSTOM_OPP_PATH" \
             "$CANN/opp/vendors/customize" \
             "$HOME/Ascend/custom_opp/vendors/customize" \
             "/usr/local/Ascend/opp/vendors/customize"; do
        if [ -n "$c" ] && [ -f "$c/op_api/include/aclnn_mul_ascend.h" ]; then
            VENDOR="$c"
            break
        fi
    done
fi
if [ -z "$VENDOR" ]; then
    c=$(ls -d "$ROOT"/build_out/_CPack_Packages/Linux/External/custom_opp_*.run/packages/vendors/customize 2>/dev/null | head -1)
    if [ -n "$c" ] && [ -f "$c/op_api/include/aclnn_mul_ascend.h" ]; then
        VENDOR="$c"
    fi
fi
if [ -z "$VENDOR" ]; then
    echo "ERROR: the MulAscend custom op is not installed (or not found)."
    echo "  1) bash build.sh"
    echo "  2) ./build_out/custom_opp_*.run --install --install-path=\$HOME/Ascend/custom_opp"
    echo "  3) export ASCEND_CUSTOM_OPP_PATH=\$HOME/Ascend/custom_opp/vendors/customize"
    echo "     (or pass --vendor <dir>/vendors/customize)"
    exit 2
fi
echo "vendor dir : $VENDOR"

INC="$VENDOR/op_api/include"
LIB="$VENDOR/op_api/lib"

echo "building   : $HERE/hw_check"
g++ -std=c++17 -O2 "$HERE/hw_check.cpp" -o "$HERE/hw_check" \
    -I"$INC" -I"$CANN/include" \
    -L"$LIB" -L"$CANN/lib64" \
    -Wl,-rpath,"$LIB" -Wl,-rpath,"$CANN/lib64" \
    -Wl,--allow-shlib-undefined \
    -lcust_opapi -lnnopbase -lascendcl -lpthread

export ASCEND_CUSTOM_OPP_PATH="$VENDOR"
export LD_LIBRARY_PATH="$LIB:$CANN/lib64:$CANN/runtime/lib64:$LD_LIBRARY_PATH"

# The runtime pulls in the driver libraries (libascend_hal.so, ...).  Add the usual locations; if they
# are missing this host simply has no NPU, and we say so instead of failing with a loader error.
for c in /usr/local/Ascend/driver/lib64 "$HOME/Ascend/driver/lib64" "$HOME/Ascend/driver/lib64/driver" \
         "$CANN/../driver/lib64"; do
    if [ -d "$c" ]; then
        export LD_LIBRARY_PATH="$c:$LD_LIBRARY_PATH"
    fi
done

if ldd "$HERE/hw_check" 2>/dev/null | grep -q 'not found'; then
    echo "ERROR: the ACL runtime cannot be loaded (missing driver libraries):"
    ldd "$HERE/hw_check" 2>/dev/null | grep 'not found' | sed 's/^/  /'
    echo "       This host has no Ascend NPU driver, so the on-device check cannot run here."
    echo "       CPU-only checks:  bash test/build_cpu_debug.sh  /  bash test/example_trace/run.sh"
    exit 3
fi

set +e
"$HERE/hw_check" "${ARGS[@]}"
rc=$?
set -e
echo "exit code  : $rc   (0 = ok, 1 = mismatch, 2 = acl/op error, 3 = no device)"
exit $rc
