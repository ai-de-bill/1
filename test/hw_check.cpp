/* ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ---------------------------------------------------------------------------------------------------------- */

/*!
 * \file hw_check.cpp
 * \brief On-device (real Ascend 910B) functional check of the MulAscend custom op through its aclnn
 *        interface: it runs the op on the NPU and compares every element with the wrap-around int64
 *        product computed on the host.  --bench additionally measures the kernel time.
 *
 * The custom op package has to be installed and visible to the runtime:
 *   ./custom_opp_ubuntu_x86_64.run --install --install-path=$HOME/Ascend/custom_opp
 *   export ASCEND_CUSTOM_OPP_PATH=$HOME/Ascend/custom_opp/vendors/customize
 *   export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
 * test/build_hw_check.sh does the build, the detection of the vendor directory and the environment
 * setup, so normally one just runs:  bash test/build_hw_check.sh [--bench [iters]] [--size N]
 *
 * exit codes: 0 = everything passed, 1 = mismatches, 2 = acl/op error, 3 = no NPU device visible.
 */

#include <acl/acl.h>
#include <aclnn/acl_meta.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "aclnn_mul_ascend.h"

namespace {

constexpr int32_t kDefaultBenchIters = 50;

// Reference: int64 multiplication with the natural wrap-around (mod 2^64).
int64_t RefMul(int64_t a, int64_t b)
{
    return static_cast<int64_t>(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
}

bool AclOk(aclError err, const char *what)
{
    if (err == ACL_SUCCESS) {
        return true;
    }
    printf("[FAIL] %s: aclError=%d (%s)\n", what, static_cast<int>(err), aclGetRecentErrMsg());
    return false;
}

struct DevTensor {
    aclTensor *tensor = nullptr;
    void *mem = nullptr;
    size_t bytes = 0;
};

bool MakeTensor(DevTensor &d, int64_t n, const std::vector<int64_t> &host, bool copyIn)
{
    d.bytes = static_cast<size_t>(n) * sizeof(int64_t);
    if (!AclOk(aclrtMalloc(&d.mem, d.bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc")) {
        return false;
    }
    if (copyIn &&
        !AclOk(aclrtMemcpy(d.mem, d.bytes, host.data(), d.bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(H2D)")) {
        return false;
    }
    const int64_t dims[1] = {n};
    const int64_t strides[1] = {1};
    d.tensor = aclCreateTensor(dims, 1, ACL_INT64, strides, 0, ACL_FORMAT_ND, dims, 1, d.mem);
    if (d.tensor == nullptr) {
        printf("[FAIL] aclCreateTensor(n=%lld) returned null\n", static_cast<long long>(n));
        return false;
    }
    return true;
}

void FreeTensor(DevTensor &d)
{
    if (d.tensor != nullptr) {
        aclDestroyTensor(d.tensor);
        d.tensor = nullptr;
    }
    if (d.mem != nullptr) {
        aclrtFree(d.mem);
        d.mem = nullptr;
    }
}

// Random values mixed with the extremes that stress the 16-bit decomposition and the wrap-around.
void FillInput(int64_t n, std::mt19937_64 &rng, std::vector<int64_t> &x, std::vector<int64_t> &y)
{
    static const int64_t kEdges[] = {0LL, 1LL, -1LL, INT64_MIN, INT64_MAX, INT64_MIN + 1, INT64_MAX - 1,
                                     0x0000000100000000LL, -0x0000000100000000LL, 0x00000000FFFFFFFFLL,
                                     static_cast<int64_t>(0xFFFFFFFF00000000ULL), 2LL, -2LL};
    const int64_t nEdges = static_cast<int64_t>(sizeof(kEdges) / sizeof(kEdges[0]));
    for (int64_t i = 0; i < n; ++i) {
        if ((i % 3) == 0) {
            x[i] = kEdges[i % nEdges];
            y[i] = kEdges[(i + 5) % nEdges];
        } else {
            x[i] = static_cast<int64_t>(rng());
            y[i] = static_cast<int64_t>(rng());
        }
    }
}

// Runs the op for one size and compares the result with the host reference.
// Returns 0 = ok, 1 = mismatch, 2 = acl/op error.
int RunOne(int64_t n, std::mt19937_64 &rng, aclrtStream stream, bool bench, int benchIters)
{
    std::vector<int64_t> hx(n), hy(n), hz(n, 0);
    FillInput(n, rng, hx, hy);

    DevTensor dx, dy, dz;
    if (!MakeTensor(dx, n, hx, true) || !MakeTensor(dy, n, hy, true) || !MakeTensor(dz, n, hz, false)) {
        FreeTensor(dx);
        FreeTensor(dy);
        FreeTensor(dz);
        return 2;
    }

    uint64_t wsSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus st = aclnnMulAscendGetWorkspaceSize(dx.tensor, dy.tensor, dz.tensor, &wsSize, &executor);
    if (st != 0 || executor == nullptr) {
        printf("[FAIL] n=%lld aclnnMulAscendGetWorkspaceSize -> %d (%s)\n", static_cast<long long>(n),
               static_cast<int>(st), aclGetRecentErrMsg());
        FreeTensor(dx);
        FreeTensor(dy);
        FreeTensor(dz);
        return 2;
    }

    void *ws = nullptr;
    if (wsSize > 0 && !AclOk(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(workspace)")) {
        aclDestroyAclOpExecutor(executor);
        FreeTensor(dx);
        FreeTensor(dy);
        FreeTensor(dz);
        return 2;
    }

    st = aclnnMulAscend(ws, wsSize, executor, stream);
    if (st != 0 || !AclOk(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
        printf("[FAIL] n=%lld aclnnMulAscend -> %d (%s)\n", static_cast<long long>(n), static_cast<int>(st),
               aclGetRecentErrMsg());
        if (ws != nullptr) {
            aclrtFree(ws);
        }
        aclDestroyAclOpExecutor(executor);
        FreeTensor(dx);
        FreeTensor(dy);
        FreeTensor(dz);
        return 2;
    }

    // Optional timing: the executor has to be made repeatable before it can be launched again.
    double usPerOp = 0.0;
    if (bench) {
        if (aclSetAclOpExecutorRepeatable(executor) != 0) {
            printf("[WARN] aclSetAclOpExecutorRepeatable failed (%s), timing is approximate\n", aclGetRecentErrMsg());
        }
        for (int32_t i = 0; i < 3; ++i) {
            (void)aclnnMulAscend(ws, wsSize, executor, stream);
        }
        (void)aclrtSynchronizeStream(stream);
        const auto t0 = std::chrono::steady_clock::now();
        for (int32_t i = 0; i < benchIters; ++i) {
            (void)aclnnMulAscend(ws, wsSize, executor, stream);
        }
        (void)aclrtSynchronizeStream(stream);
        const auto t1 = std::chrono::steady_clock::now();
        usPerOp = std::chrono::duration<double, std::micro>(t1 - t0).count() / benchIters;
    }

    if (!AclOk(aclrtMemcpy(hz.data(), dz.bytes, dz.mem, dz.bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(D2H)")) {
        if (ws != nullptr) {
            aclrtFree(ws);
        }
        aclDestroyAclOpExecutor(executor);
        FreeTensor(dx);
        FreeTensor(dy);
        FreeTensor(dz);
        return 2;
    }

    int64_t bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t ref = RefMul(hx[i], hy[i]);
        if (hz[i] != ref) {
            if (bad < 3) {
                printf("   mismatch i=%lld x=%lld y=%lld ref=%lld got=%lld\n", static_cast<long long>(i),
                       static_cast<long long>(hx[i]), static_cast<long long>(hy[i]), static_cast<long long>(ref),
                       static_cast<long long>(hz[i]));
            }
            ++bad;
        }
    }

    if (bench) {
        printf("  n=%-8lld %-6s (%.2f us/op, %.3f ns/element, %d iters)\n", static_cast<long long>(n),
               (bad == 0) ? "OK" : "FAILED", usPerOp,
               (n > 0) ? (usPerOp * 1000.0 / static_cast<double>(n)) : 0.0, benchIters);
    } else {
        printf("  n=%-8lld %-6s%s\n", static_cast<long long>(n), (bad == 0) ? "OK" : "FAILED",
               (bad == 0) ? "" : "  <-- wrong results");
    }

    if (ws != nullptr) {
        aclrtFree(ws);
    }
    aclDestroyAclOpExecutor(executor);
    FreeTensor(dx);
    FreeTensor(dy);
    FreeTensor(dz);
    return (bad == 0) ? 0 : 1;
}

} // namespace


int main(int argc, char **argv)
{
    bool bench = false;
    int32_t benchIters = kDefaultBenchIters;
    int64_t singleSize = -1;

    for (int32_t i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) {
            bench = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                benchIters = std::atoi(argv[++i]);
            }
        } else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            singleSize = std::atoll(argv[++i]);
        } else {
            printf("usage: %s [--bench [iters]] [--size N]\n", argv[0]);
            return 2;
        }
    }

    if (!AclOk(aclInit(nullptr), "aclInit")) {
        return 2;
    }

    uint32_t devCount = 0;
    if (!AclOk(aclrtGetDeviceCount(&devCount), "aclrtGetDeviceCount") || devCount == 0) {
        printf("[SKIP] no Ascend device is visible (count=%u).\n", devCount);
        printf("       This check needs real hardware.  For the CPU debug model use:\n");
        printf("         bash test/build_cpu_debug.sh     (22 sizes on the 910B functional model)\n");
        printf("         bash test/example_trace/run.sh   (dumps every intermediate buffer)\n");
        aclFinalize();
        return 3;
    }

    if (!AclOk(aclrtSetDevice(0), "aclrtSetDevice")) {
        aclFinalize();
        return 2;
    }
    aclrtStream stream = nullptr;
    if (!AclOk(aclrtCreateStream(&stream), "aclrtCreateStream")) {
        aclrtResetDevice(0);
        aclFinalize();
        return 2;
    }

    std::vector<int64_t> sizes;
    if (singleSize > 0) {
        sizes.push_back(singleSize);
    } else {
        sizes = {1, 2, 3, 5, 7, 15, 16, 17, 255, 256, 257, 511, 512, 513, 1000, 4095, 4096, 4097, 65536, 100000,
                 1 << 20};
    }

    printf("mul_ascend device check: %u device(s) visible, %zu size(s), bench=%s\n", devCount, sizes.size(),
           bench ? "on" : "off");

    std::mt19937_64 rng(20251018ULL);
    int32_t failed = 0;
    int32_t errored = 0;
    for (int64_t n : sizes) {
        const int32_t rc = RunOne(n, rng, stream, bench, benchIters);
        if (rc == 1) {
            ++failed;
        } else if (rc == 2) {
            ++errored;
            break; // acl/op level error: the remaining sizes would fail in the same way
        }
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    if (errored != 0) {
        printf("FAILED: acl/op error\n");
        return 2;
    }
    if (failed != 0) {
        printf("FAILED: %d size(s) produced wrong results\n", failed);
        return 1;
    }
    printf("ALL PASSED (%zu sizes)\n", sizes.size());
    return 0;
}

