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
 * TilingFunc: a uint32 element count at offset 0 (op_host/mul_ascend.cpp).  The shape is therefore
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
#include <dlfcn.h>
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
    aclrtBinHandle bin = nullptr;   // aclrtBinaryLoadFromFile handle (acl path)
    aclrtFuncHandle func = nullptr; // resolved kernel entry (acl path)
    void *rtHandle = nullptr;       // rtRegisterAllKernel handle (runtime path)
    std::vector<char> binData;      // the .o bytes, needed by rtRegisterAllKernel
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

/* The runtime hands the kernel its arguments as one blob.  For a kernel built by the op framework the
 * tiling data travels *inline* in that blob - that is what rtArgsEx_t::tilingDataOffset describes and what
 * opParaSize (12 for this op) accounts for: the tiling parameter of the kernel is patched by the runtime to
 * point at the inline copy.  So aclrtLaunchKernel gets 5 pointers (40 bytes) followed by the tiling data.
 * kTilingInlineBytes is 16 because the inline area is kept 16-byte aligned. */
constexpr size_t kTilingInlineBytes = 16;

/* ---- low level runtime API ----
 * A .o produced by the custom-op build is registered and launched the way the official MindStudio "kernel
 * launch" tool (mskl) does it: rtRegisterAllKernel() + rtKernelLaunchWithHandleV2().  The runtime then takes
 * care of the argument blob (args pointers are copied to the device by the runtime itself) and of the
 * tiling parameter of the kernel; hasTiling = 0 says the tiling is a plain pointer argument. */
typedef struct RtDevBinary {
    uint32_t magic;    // 0x41415246 == RT_DEV_BINARY_MAGIC_ELF_AIVEC (dav-c220 vector kernel)
    uint32_t version;
    const void *data;
    uint64_t length;
} RtDevBinary;

typedef struct RtHostInputInfo {
    uint32_t addrOffset;
    uint32_t dataOffset;
} RtHostInputInfo;

typedef struct RtArgsEx {
    void *args;
    RtHostInputInfo *hostInputInfoPtr;
    uint32_t argsSize;
    uint32_t tilingAddrOffset;
    uint32_t tilingDataOffset;
    uint16_t hostInputInfoNum;
    uint8_t hasTiling;
    uint8_t isNoNeedH2DCopy;
    uint8_t reserved[4];
} RtArgsEx;

constexpr uint32_t kMagicElfAivec = 0x41415246U;

struct RtApi {
    int (*registerAllKernel)(const RtDevBinary *, void **) = nullptr;
    int (*launchWithHandleV2)(void *, uint64_t, uint32_t, RtArgsEx *, void *, void *, const void *) = nullptr;
    int (*devBinaryUnRegister)(void *) = nullptr;
    std::string missing;

    bool Load()
    {
        void *lib = dlopen("libruntime.so", RTLD_LAZY | RTLD_GLOBAL);
        registerAllKernel = reinterpret_cast<decltype(registerAllKernel)>(dlsym(RTLD_DEFAULT, "rtRegisterAllKernel"));
        launchWithHandleV2 =
            reinterpret_cast<decltype(launchWithHandleV2)>(dlsym(RTLD_DEFAULT, "rtKernelLaunchWithHandleV2"));
        devBinaryUnRegister =
            reinterpret_cast<decltype(devBinaryUnRegister)>(dlsym(RTLD_DEFAULT, "rtDevBinaryUnRegister"));
        if (registerAllKernel == nullptr || launchWithHandleV2 == nullptr || devBinaryUnRegister == nullptr) {
            if (lib == nullptr) {
                missing = dlerror() != nullptr ? dlerror() : "libruntime.so not loadable";
            } else {
                missing = "rtRegisterAllKernel / rtKernelLaunchWithHandleV2 not found";
            }
            return false;
        }
        return true;
    }
};

// Everything the launcher needs to know about how the kernel is started.
struct LaunchConfig {
    const KernelHandle *kernel = nullptr;
    const RtApi *rt = nullptr;   // nullptr -> aclrtLaunchKernel, otherwise rtKernelLaunchWithHandleV2
    uint64_t tilingKey = 0;
    uint32_t blockDim = kBlockDim;
};

/* The kernel takes five GM_ADDR arguments, i.e. 40 bytes of kernel arguments. */
bool Launch(const LaunchConfig &lc, aclrtStream stream, void *const *args, uint32_t elems)
{
    if (lc.rt != nullptr) {
        RtArgsEx argsEx{};
        argsEx.args = const_cast<void **>(args);
        argsEx.hostInputInfoPtr = nullptr;
        argsEx.argsSize = sizeof(void *) * kArgCount;
        argsEx.hasTiling = 0;
        argsEx.isNoNeedH2DCopy = 0;
        const int32_t ret = lc.rt->launchWithHandleV2(lc.kernel->rtHandle, lc.tilingKey, lc.blockDim, &argsEx,
                                                      nullptr, stream, nullptr);
        if (ret != 0) {
            printf("[FAIL] rtKernelLaunchWithHandleV2: ret=%d\n", ret);
            return false;
        }
        return true;
    }

    // aclrtLaunchKernel path: the arguments travel as one blob.  For a kernel built by the op framework the
    // tiling data is part of that blob (rtArgsEx_t::tilingDataOffset, "opParaSize" = 12 for this op), so the
    // five pointers are followed by the tiling data itself.
    std::vector<uint8_t> argBlob(sizeof(void *) * kArgCount + kTilingInlineBytes, 0);
    std::memcpy(argBlob.data(), args, sizeof(void *) * kArgCount);
    std::memcpy(argBlob.data() + sizeof(void *) * kArgCount, &elems, sizeof(elems));
    return AclOk(aclrtLaunchKernel(lc.kernel->func, lc.blockDim, argBlob.data(), argBlob.size(), stream),
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

// rtRegisterAllKernel() takes the binary from host memory, so the .o is read here.
bool ReadFile(const std::string &path, std::vector<char> &data)
{
    FILE *fd = std::fopen(path.c_str(), "rb");
    if (fd == nullptr) {
        printf("[FAIL] cannot open '%s'\n", path.c_str());
        return false;
    }
    std::fseek(fd, 0, SEEK_END);
    const long size = std::ftell(fd);
    std::fseek(fd, 0, SEEK_SET);
    if (size <= 0) {
        std::fclose(fd);
        printf("[FAIL] '%s' is empty\n", path.c_str());
        return false;
    }
    data.resize(static_cast<size_t>(size));
    const size_t got = std::fread(data.data(), 1, data.size(), fd);
    std::fclose(fd);
    if (got != data.size()) {
        printf("[FAIL] short read of '%s' (%zu of %zu bytes)\n", path.c_str(), got, data.size());
        return false;
    }
    return true;
}

int32_t RunOne(int64_t n, std::mt19937_64 &rng, aclrtStream stream, const LaunchConfig &lc, bool bench,
               int32_t benchIters, bool checkInputs)
{
    const size_t bytes = static_cast<size_t>(n) * sizeof(int64_t);
    // The output is pre-filled with a sentinel: elements that still hold it were never written by the
    // kernel, which tells "no work done" apart from "wrong value computed".
    const int64_t kSentinel = static_cast<int64_t>(0x5a5a5a5a5a5a5a5aULL);
    std::vector<int64_t> hx(static_cast<size_t>(n));
    std::vector<int64_t> hy(static_cast<size_t>(n));
    std::vector<int64_t> hz(static_cast<size_t>(n), kSentinel);

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

    if (checkInputs) { // read x back: proves the host-to-device copies really landed
        std::vector<int64_t> back(static_cast<size_t>(n), 0);
        if (!AclOk(aclrtMemcpy(back.data(), bytes, dx.mem, bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                   "aclrtMemcpy(x back)")) {
            return 2;
        }
        const bool same = (std::memcmp(back.data(), hx.data(), bytes) == 0);
        printf("  n=%-8lld inputs %s (device copy of x read back)\n", static_cast<long long>(n),
               same ? "OK" : "MISMATCH");
        if (!same) {
            printf("           device x[0]=0x%016llx  host x[0]=0x%016llx\n",
                   static_cast<unsigned long long>(back[0]), static_cast<unsigned long long>(hx[0]));
        }
    }

    // Tiling: what TilingFunc would have produced - the element count at offset 0.  The whole buffer is
    // filled with that value: the tiling struct is a single uint32, so any padding/offset read by the
    // generated GET_TILING_DATA still sees the element count.
    std::vector<uint8_t> hTiling(kTilingBytes, 0);
    const uint32_t elems = static_cast<uint32_t>(n);
    for (size_t off = 0; off + sizeof(elems) <= kTilingBytes; off += sizeof(elems)) {
        std::memcpy(hTiling.data() + off, &elems, sizeof(elems));
    }
    if (!AclOk(aclrtMemcpy(dTiling.mem, kTilingBytes, hTiling.data(), kTilingBytes, ACL_MEMCPY_HOST_TO_DEVICE),
               "aclrtMemcpy(tiling)")) {
        return 2;
    }

    void *args[kArgCount] = {dx.mem, dy.mem, dz.mem, dWs.mem, dTiling.mem};

    if (checkInputs) { // diagnostics: device addresses (useful with the simulator logs) and the H2D check
        printf("  n=%-8lld dev x=%p y=%p z=%p ws=%p tiling=%p\n", static_cast<long long>(n), dx.mem, dy.mem,
               dz.mem, dWs.mem, dTiling.mem);
    }

    double usPerOp = 0.0;
    if (bench) {
        for (int32_t i = 0; i < 3; ++i) { // warm-up
            if (!Launch(lc, stream, args, elems)) {
                return 2;
            }
        }
        if (!AclOk(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(warmup)")) {
            return 2;
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int32_t i = 0; i < benchIters; ++i) {
            if (!Launch(lc, stream, args, elems)) {
                return 2;
            }
        }
        if (!AclOk(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(bench)")) {
            return 2;
        }
        const auto t1 = std::chrono::steady_clock::now();
        usPerOp = std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(benchIters);
    } else if (!Launch(lc, stream, args, elems)) {
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
    int64_t untouched = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (hz[static_cast<size_t>(i)] == kSentinel) {
            ++untouched;
        }
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
        const int64_t i = firstBad;
        printf("           %lld mismatching element(s), first at index %lld\n", static_cast<long long>(bad),
               static_cast<long long>(firstBad));
        printf("           x=%lld (0x%016llx) y=%lld (0x%016llx)\n",
               static_cast<long long>(hx[static_cast<size_t>(i)]),
               static_cast<unsigned long long>(hx[static_cast<size_t>(i)]),
               static_cast<long long>(hy[static_cast<size_t>(i)]),
               static_cast<unsigned long long>(hy[static_cast<size_t>(i)]));
        printf("           got=0x%016llx  expected=0x%016llx\n",
               static_cast<unsigned long long>(hz[static_cast<size_t>(i)]),
               static_cast<unsigned long long>(RefMul(hx[static_cast<size_t>(i)], hy[static_cast<size_t>(i)])));
        if (untouched == n) {
            printf("           the kernel never wrote the output: all %lld element(s) still hold the sentinel\n",
                   static_cast<long long>(n));
        } else if (untouched != 0) {
            printf("           %lld of %lld element(s) were never written\n", static_cast<long long>(untouched),
                   static_cast<long long>(n));
        }
        const int64_t show = (n < 8) ? n : 8;
        printf("           got     :");
        for (int64_t i = 0; i < show; ++i) {
            printf(" %016llx", static_cast<unsigned long long>(hz[static_cast<size_t>(i)]));
        }
        printf("\n           expected:");
        for (int64_t i = 0; i < show; ++i) {
            printf(" %016llx",
                   static_cast<unsigned long long>(RefMul(hx[static_cast<size_t>(i)], hy[static_cast<size_t>(i)])));
        }
        printf("\n");
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
    printf("  --entry N       tiling key (aclrtBinaryGetFunctionByEntry / rtKernelLaunchWithHandleV2, default: 0)\n");
    printf("  --launch-api A  rt = rtRegisterAllKernel + rtKernelLaunchWithHandleV2 (default),\n");
    printf("                  acl = aclrtBinaryLoadFromFile + aclrtLaunchKernel\n");
    printf("  --magic N       binary magic for the rt path (default: 0x%08x = RT_DEV_BINARY_MAGIC_ELF_AIVEC)\n",
           kMagicElfAivec);
    printf("  --block-dim N   number of blocks, must match the kernel (default: %u)\n", kBlockDim);
    printf("  --device N      device id (default: 0)\n");
    printf("  --size N        run a single size instead of the default list\n");
    printf("  --bench [iters] measure the kernel time as well (default iters: %d)\n", kDefaultBenchIters);
    printf("  --check-inputs  read x back from the device to verify the host-to-device copies\n");
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
    bool checkInputs = false;
    int32_t benchIters = kDefaultBenchIters;
    std::string launchApi = "rt";           // rt = rtRegisterAllKernel + rtKernelLaunchWithHandleV2, acl = aclrtLaunchKernel
    uint32_t magic = kMagicElfAivec;

    for (int32_t i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            kernelPath = argv[++i];
        } else if (std::strcmp(argv[i], "--name") == 0 && i + 1 < argc) {
            kernelName = argv[++i];
        } else if (std::strcmp(argv[i], "--entry") == 0 && i + 1 < argc) {
            entry = std::strtoull(argv[++i], nullptr, 0);
        } else if (std::strcmp(argv[i], "--launch-api") == 0 && i + 1 < argc) {
            launchApi = argv[++i];
        } else if (std::strcmp(argv[i], "--magic") == 0 && i + 1 < argc) {
            magic = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 0));
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
        } else if (std::strcmp(argv[i], "--check-inputs") == 0) {
            checkInputs = true;
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
    RtApi rt;
    LaunchConfig lc;
    lc.kernel = &kh;
    lc.blockDim = blockDim;
    lc.tilingKey = entry;

    if (launchApi == "rt") {
        if (!rt.Load()) {
            printf("[FAIL] the runtime API is not available: %s\n", rt.missing.c_str());
            aclrtDestroyStream(stream);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return 2;
        }
        if (!ReadFile(kernelPath, kh.binData)) {
            aclrtDestroyStream(stream);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return 2;
        }
        RtDevBinary bin{};
        bin.magic = magic;
        bin.version = 0;
        bin.data = kh.binData.data();
        bin.length = static_cast<uint64_t>(kh.binData.size());
        const int32_t ret = rt.registerAllKernel(&bin, &kh.rtHandle);
        if (ret != 0) {
            printf("[FAIL] rtRegisterAllKernel: ret=%d (magic=0x%08x, %zu bytes)\n", ret, magic, kh.binData.size());
            aclrtDestroyStream(stream);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return 2;
        }
        lc.rt = &rt;
        char magicStr[16] = {0};
        std::snprintf(magicStr, sizeof(magicStr), "0x%08x", magic);
        kh.resolved = std::string("rtRegisterAllKernel (magic=") + magicStr +
                      ") + rtKernelLaunchWithHandleV2(tilingKey=" + std::to_string(entry) + ")";
    } else {
        if (!LoadKernel(kernelPath.c_str(), kernelName.c_str(), entry, kh)) {
            aclrtDestroyStream(stream);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return 2;
        }
        kh.resolved = "aclrtBinaryGetFunction " + kh.resolved;
    }
    printf("launch     : %s\n", kh.resolved.c_str());

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
        const int32_t rc = RunOne(n, rng, stream, lc, bench, benchIters, checkInputs);
        if (rc == 1) {
            ++failed;
        } else if (rc == 2) {
            ++errored;
            break; // acl level error: the remaining sizes would fail the same way
        }
    }

    if (kh.rtHandle != nullptr) {
        rt.devBinaryUnRegister(kh.rtHandle);
    } else {
        aclrtBinaryUnLoad(kh.bin);
    }
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
