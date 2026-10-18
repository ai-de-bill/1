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
 * \file direct_launch_check.cpp
 * \brief Launch the MulAscend kernel WITHOUT registering the operator in CANN.
 *
 * The device .o built by `bash build.sh` is loaded straight from disk with aclrtBinaryLoadFromFile, the
 * kernel entry is resolved with aclrtBinaryGetFunction (falling back to aclrtBinaryGetFunctionByEntry) and
 * the launch is done with aclrtLaunchKernel.  Nothing of the custom-op framework takes part at run time:
 * no aclnn wrapper, no TilingFunc call, no custom_opp_*.run package and no ASCEND_CUSTOM_OPP_PATH - the
 * .o is used as it is (it already carries its .ascend.meta.<kernel> section, i.e. the kernel type and the
 * argument description the driver reads).
 *
 * Kernel arguments are exactly the five GM_ADDR of the kernel entry:
 *
 *     mul_ascend(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
 *
 * `tiling` points at a device buffer that holds the tiling struct the op would normally get from its
 * TilingFunc: a uint32 element count at offset 0 (op_kernel/mul_ascend_tiling.h).  The shape is therefore
 * chosen here at run time, so arbitrary sizes can be checked.
 *
 * Build/run: bash test/build_direct_launch.sh [--size N] [--bench [iters]] [--block-dim N]
 *
 * exit codes: 0 = everything passed, 1 = mismatches, 2 = acl error, 3 = no NPU device visible.
 */

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int32_t kDefaultBenchIters = 50;
constexpr uint32_t kBlockDim = 16;       // the grid the kernel was written for (TilingFunc: SetBlockDim(16))
constexpr uint32_t kTilingBytes = 32;    // the tiling struct is 4 bytes, the op declares 12 - keep headroom
constexpr uint32_t kWorkspaceBytes = 32; // mul_ascend ignores its workspace, but a real pointer is passed
constexpr int32_t kArgCount = 5;         // x, y, z, workspace, tiling

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

struct KernelHandle {
    aclrtBinHandle bin = nullptr;
    aclrtFuncHandle func = nullptr;
    std::string resolved;
};

/* Load the device .o and resolve the kernel entry: by name first, then by tiling-key entry. */
bool LoadKernel(const char *path, const char *name, uint64_t entry, KernelHandle &kh)
{
    if (!AclOk(aclrtBinaryLoadFromFile(path, nullptr, &kh.bin), "aclrtBinaryLoadFromFile")) {
        return false;
    }
    if (name != nullptr && name[0] != '\0') {
        if (aclrtBinaryGetFunction(kh.bin, name, &kh.func) == ACL_SUCCESS) {
            kh.resolved = std::string("name '") + name + "'";
            return true;
        }
        printf("[WARN] aclrtBinaryGetFunction('%s') failed (%s); trying entry %llu\n", name,
               aclGetRecentErrMsg(), static_cast<unsigned long long>(entry));
    }
    if (!AclOk(aclrtBinaryGetFunctionByEntry(kh.bin, entry, &kh.func), "aclrtBinaryGetFunctionByEntry")) {
        return false;
    }
    kh.resolved = "entry " + std::to_string(entry);
    return true;
}

/* The kernel takes five GM_ADDR arguments, i.e. 40 bytes of kernel arguments, passed as one flat buffer. */
bool LaunchKernel(const KernelHandle &kh, uint32_t blockDim, aclrtStream stream, void *const *args)
{
    return AclOk(aclrtLaunchKernel(kh.func, blockDim, args, sizeof(void *) * kArgCount, stream),
                 "aclrtLaunchKernel");
}

struct DeviceBuffer {
    void *mem = nullptr;

    bool Alloc(size_t bytes)
    {
        return AclOk(aclrtMalloc(&mem, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }
    ~DeviceBuffer()
    {
        if (mem != nullptr) {
            aclrtFree(mem);
        }
    }
};

int32_t RunOne(int64_t n, std::mt19937_64 &rng, aclrtStream stream, const KernelHandle &kh, bool bench,
               int32_t benchIters)
{
    const size_t bytes = static_cast<size_t>(n) * sizeof(int64_t);
    std::vector<int64_t> hx(static_cast<size_t>(n));
    std::vector<int64_t> hy(static_cast<size_t>(n));
    std::vector<int64_t> hz(static_cast<size_t>(n), 0);

    for (int64_t i = 0; i < n; ++i) {
        hx[static_cast<size_t>(i)] = static_cast<int64_t>(rng());
        hy[static_cast<size_t>(i)] = static_cast<int64_t>(rng());
        if ((i % 7) == 0) { // mix in extremes
            hx[static_cast<size_t>(i)] = (i % 14 == 0) ? INT64_MIN : INT64_MAX;
            hy[static_cast<size_t>(i)] = (i % 14 == 0) ? INT64_MAX : INT64_MIN;
        }
    }

    DeviceBuffer dx;
    DeviceBuffer dy;
    DeviceBuffer dz;
    DeviceBuffer dWs;
    DeviceBuffer dTiling;
    if (!dx.Alloc(bytes) || !dy.Alloc(bytes) || !dz.Alloc(bytes) || !dWs.Alloc(kWorkspaceBytes) ||
        !dTiling.Alloc(kTilingBytes)) {
        return 2;
    }

    if (!AclOk(aclrtMemcpy(dx.mem, bytes, hx.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(x)") ||
        !AclOk(aclrtMemcpy(dy.mem, bytes, hy.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(y)") ||
        !AclOk(aclrtMemcpy(dz.mem, bytes, hz.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(z)")) {
        return 2;
    }

    // Tiling: what TilingFunc would have produced - the element count at offset 0.
    std::vector<uint8_t> hTiling(kTilingBytes, 0);
    const uint32_t elems = static_cast<uint32_t>(n);
    std::memcpy(hTiling.data(), &elems, sizeof(elems));
    if (!AclOk(aclrtMemcpy(dTiling.mem, kTilingBytes, hTiling.data(), kTilingBytes, ACL_MEMCPY_HOST_TO_DEVICE),
               "aclrtMemcpy(tiling)")) {
        return 2;
    }

    void *args[kArgCount] = {dx.mem, dy.mem, dz.mem, dWs.mem, dTiling.mem};

    double usPerOp = 0.0;
    if (bench) {
        for (int32_t i = 0; i < 3; ++i) { // warm-up
            if (!LaunchKernel(kh, kBlockDim, stream, args)) {
                return 2;
            }
        }
        if (!AclOk(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(warmup)")) {
            return 2;
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int32_t i = 0; i < benchIters; ++i) {
            if (!LaunchKernel(kh, kBlockDim, stream, args)) {
                return 2;
            }
        }
        if (!AclOk(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(bench)")) {
            return 2;
        }
        const auto t1 = std::chrono::steady_clock::now();
        usPerOp = std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(benchIters);
    } else if (!LaunchKernel(kh, kBlockDim, stream, args)) {
        return 2;
    }

    if (!bench && !AclOk(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
        return 2;
    }
    if (!AclOk(aclrtMemcpy(hz.data(), bytes, dz.mem, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(z out)")) {
        return 2;
    }

    int64_t bad = 0;
    int64_t firstBad = -1;
    for (int64_t i = 0; i < n; ++i) {
        if (hz[static_cast<size_t>(i)] != RefMul(hx[static_cast<size_t>(i)], hy[static_cast<size_t>(i)])) {
            if (bad == 0) {
                firstBad = i;
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
    if (bad != 0) {
        printf("           %lld mismatching element(s), first at index %lld\n", static_cast<long long>(bad),
               static_cast<long long>(firstBad));
    }
    return (bad == 0) ? 0 : 1;
}

} // namespace

void Usage(const char *prog)
{
    printf("usage: %s --kernel PATH [--name KERNEL] [--entry N] [--block-dim N] [--device N]\n", prog);
    printf("          [--size N] [--bench [iters]]\n\n");
    printf("  --kernel PATH   device .o to load (e.g. build_out/.../MulAscend_<hash>.o)\n");
    printf("  --name KERNEL   kernel name for aclrtBinaryGetFunction (default: from the .o symbol)\n");
    printf("  --entry N       tiling-key entry for aclrtBinaryGetFunctionByEntry (default: 0)\n");
    printf("  --block-dim N   number of blocks, must match the kernel (default: %u)\n", kBlockDim);
    printf("  --device N      device id (default: 0)\n");
    printf("  --size N        run a single size instead of the default list\n");
    printf("  --bench [iters] measure the kernel time as well (default iters: %d)\n", kDefaultBenchIters);
}

int main(int argc, char **argv)
{
    std::string kernelPath;
    std::string kernelName;
    uint64_t entry = 0;
    uint32_t blockDim = kBlockDim;
    int32_t deviceId = 0;
    int64_t singleSize = -1;
    bool bench = false;
    int32_t benchIters = kDefaultBenchIters;

    for (int32_t i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            kernelPath = argv[++i];
        } else if (std::strcmp(argv[i], "--name") == 0 && i + 1 < argc) {
            kernelName = argv[++i];
        } else if (std::strcmp(argv[i], "--entry") == 0 && i + 1 < argc) {
            entry = std::strtoull(argv[++i], nullptr, 0);
        } else if (std::strcmp(argv[i], "--block-dim") == 0 && i + 1 < argc) {
            blockDim = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            deviceId = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            singleSize = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--bench") == 0) {
            bench = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                benchIters = std::atoi(argv[++i]);
            }
        } else {
            Usage(argv[0]);
            return 2;
        }
    }
    if (kernelPath.empty()) {
        printf("ERROR: --kernel PATH is required (use bash test/build_direct_launch.sh).\n\n");
        Usage(argv[0]);
        return 2;
    }

    if (!AclOk(aclInit(nullptr), "aclInit")) {
        return 2;
    }

    uint32_t devCount = 0;
    if (!AclOk(aclrtGetDeviceCount(&devCount), "aclrtGetDeviceCount") || devCount == 0) {
        printf("[SKIP] no Ascend device is visible (count=%u).\n", devCount);
        printf("       The direct launch needs real hardware.  CPU-only checks:\n");
        printf("         bash test/build_cpu_debug.sh    (22 sizes on the 910B functional model)\n");
        printf("         bash test/example_trace/run.sh  (dumps every intermediate buffer)\n");
        aclFinalize();
        return 3;
    }
    if (!AclOk(aclrtSetDevice(deviceId), "aclrtSetDevice")) {
        aclFinalize();
        return 2;
    }
    aclrtStream stream = nullptr;
    if (!AclOk(aclrtCreateStream(&stream), "aclrtCreateStream")) {
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 2;
    }

    printf("direct launch (no op registration): %u device(s), block-dim=%u\n", devCount, blockDim);
    printf("kernel .o  : %s\n", kernelPath.c_str());

    KernelHandle kh;
    if (!LoadKernel(kernelPath.c_str(), kernelName.c_str(), entry, kh)) {
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 2;
    }
    printf("entry      : %s\n", kh.resolved.c_str());

    std::vector<int64_t> sizes;
    if (singleSize > 0) {
        sizes.push_back(singleSize);
    } else {
        sizes = {1, 2, 3, 5, 7, 15, 16, 17, 255, 256, 257, 511, 512, 513, 1000, 4095, 4096, 4097, 65536, 100000,
                 1 << 20};
    }

    std::mt19937_64 rng(20251018ULL);
    int32_t failed = 0;
    int32_t errored = 0;
    for (int64_t n : sizes) {
        const int32_t rc = RunOne(n, rng, stream, kh, bench, benchIters);
        if (rc == 1) {
            ++failed;
        } else if (rc == 2) {
            ++errored;
            break; // acl level error: the remaining sizes would fail the same way
        }
    }

    aclrtBinaryUnLoad(kh.bin);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (errored != 0) {
        printf("FAILED: acl error\n");
        return 2;
    }
    if (failed != 0) {
        printf("FAILED: %d size(s) produced wrong results\n", failed);
        return 1;
    }
    printf("ALL PASSED (%zu sizes, direct kernel launch)\n", sizes.size());
    return 0;
}
