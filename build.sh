#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

if [ -z "$BASE_LIBS_PATH" ]; then
    if [ -z "$ASCEND_HOME_PATH" ]; then
        if [ -z "$ASCEND_AICPU_PATH" ]; then
            # Auto-detect common install locations to make build.sh self-contained.
            if [ -d "/home/user/Ascend/ascend-toolkit/cann-9.1.0" ]; then
                export ASCEND_HOME_PATH="/home/user/Ascend/ascend-toolkit/cann-9.1.0"
            elif [ -d "/usr/local/Ascend/ascend-toolkit/latest" ]; then
                export ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest"
            elif [ -d "/usr/local/Ascend/latest" ]; then
                export ASCEND_HOME_PATH="/usr/local/Ascend/latest"
            else
                echo "please set env."
                exit 1
            fi
        else
            export ASCEND_HOME_PATH=$ASCEND_AICPU_PATH
        fi
    else
        export ASCEND_HOME_PATH=$ASCEND_HOME_PATH
    fi
else
    export ASCEND_HOME_PATH=$BASE_LIBS_PATH
fi
echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
script_path=$(realpath $(dirname $0))

# Toolkit runtime libs required by opbuild/op compiler tools.
append_ld_path() {
    if [ -d "$1" ]; then
        if [ -n "$LD_LIBRARY_PATH" ]; then
            export LD_LIBRARY_PATH="$1:$LD_LIBRARY_PATH"
        else
            export LD_LIBRARY_PATH="$1"
        fi
    fi
}

append_ld_path "$ASCEND_HOME_PATH/x86_64-linux/lib64"
append_ld_path "$ASCEND_HOME_PATH/runtime/lib64"
append_ld_path "$ASCEND_HOME_PATH/compiler/lib64"
append_ld_path "$ASCEND_HOME_PATH/tools/opbuild"

# Required compiler tools (asc_opc, ccec, etc.)
append_path() {
    if [ -d "$1" ]; then
        if [ -n "$PATH" ]; then
            export PATH="$1:$PATH"
        else
            export PATH="$1"
        fi
    fi
}

append_path "$ASCEND_HOME_PATH/x86_64-linux/bin"
append_path "$ASCEND_HOME_PATH/compiler/bin"

# Resolve ASC package path for find_package(ASC REQUIRED).
if [ -z "$ASC_DIR" ]; then
    if [ -f "$ASCEND_HOME_PATH/x86_64-linux/lib64/cmake/asc-config.cmake" ]; then
        export ASC_DIR="$ASCEND_HOME_PATH/x86_64-linux/lib64/cmake"
    elif [ -f "$ASCEND_HOME_PATH/aarch64-linux/lib64/cmake/asc-config.cmake" ]; then
        export ASC_DIR="$ASCEND_HOME_PATH/aarch64-linux/lib64/cmake"
    elif [ -f "$ASCEND_HOME_PATH/lib64/cmake/asc-config.cmake" ]; then
        export ASC_DIR="$ASCEND_HOME_PATH/lib64/cmake"
    fi
fi

if [ -n "$ASC_DIR" ]; then
    if [ -n "$CMAKE_PREFIX_PATH" ]; then
        export CMAKE_PREFIX_PATH="$ASC_DIR:$CMAKE_PREFIX_PATH"
    else
        export CMAKE_PREFIX_PATH="$ASC_DIR"
    fi
fi

BUILD_DIR="build_out"
mkdir -p build_out
rm -rf build_out/*
opts=$(python3 $ASCEND_HOME_PATH/tools/tikcpp/ascendc_kernel_cmake/fwk_modules/util/preset_parse.py $script_path/CMakePresets.json)
cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')

if [ "$cmake_version" \< "3.19.0" ] ; then
    cmake -S . -B "$BUILD_DIR" $opts -DASC_DIR="$ASC_DIR" -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"
else
    cmake -S . -B "$BUILD_DIR" --preset=default -DASC_DIR="$ASC_DIR" -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"
fi
cmake --build "$BUILD_DIR" --target binary package -j$(nproc)
