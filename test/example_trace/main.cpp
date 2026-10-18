// Test-only harness: runs op_kernel/mul_ascend.cpp (instrumented copy) on the Ascend910B CPU debug
// model with a hand-picked 3-element input and dumps every intermediate buffer, so that the data
// transformation can be visualised with real values.
#include <cstdint>
#include <cstdio>

#include "mul_ascend_tiling.h"

static uint32_t g_tilingSize = 3;
#define REGISTER_TILING_DEFAULT(tilingStruct)
#define GET_TILING_DATA(var, tilingArg) \
    MulAscendTilingData var;            \
    var.size = g_tilingSize

#include "mul_ascend_dbg.cpp"   // instrumented copy of the kernel under test

#include "tikicpulib.h"

// 3 elements: a trivial one, the maximal-carry one, and a "random" one.
static const uint64_t EX_X[3] = {0x0000000200000001ULL, 0xFFFFFFFFFFFFFFFFULL, 0x123456789ABCDEF0ULL};
static const uint64_t EX_Y[3] = {0x0000000300000004ULL, 0xFFFFFFFFFFFFFFFFULL, 0xFEDCBA9876543210ULL};

static const char* SLOT_NAME[32] = {
    "tLoBuf = [aLo|bLo] : aLo", "tLoBuf = [aLo|bLo] : bLo", "tHiBuf = [aHi|bHi] : aHi",
    "tHiBuf = [aHi|bHi] : bHi", "in32 (raw tile lanes)",    "tPack : rLo", "tPack : rHi", "-", "MulHi: x0",
    "MulHi: y0", "MulHi: x1", "MulHi: y1", "MulHi: U=x0*y1", "MulHi: V=x1*y0", "MulHi: W=x1*y1",
    "MulHi: A=x0*y0", "MulHi: U0", "MulHi: V0", "MulHi: s0(U0+V0)", "MulHi: c0=s0>>16",
    "MulHi: T_hi", "MulHi: S_hi", nullptr, nullptr};

int main()
{
    uint8_t* x = static_cast<uint8_t*>(AscendC::GmAlloc(4096));
    uint8_t* y = static_cast<uint8_t*>(AscendC::GmAlloc(4096));
    uint8_t* z = static_cast<uint8_t*>(AscendC::GmAlloc(4096));
    uint8_t* ws = static_cast<uint8_t*>(AscendC::GmAlloc(8));
    uint8_t* tl = static_cast<uint8_t*>(AscendC::GmAlloc(8));
    uint32_t* dump = static_cast<uint32_t*>(AscendC::GmAlloc(4096));

    for (int i = 0; i < 512; ++i) { dump[i] = 0xDEADBEEF; }
    reinterpret_cast<uint64_t*>(x)[0] = EX_X[0];
    reinterpret_cast<uint64_t*>(x)[1] = EX_X[1];
    reinterpret_cast<uint64_t*>(x)[2] = EX_X[2];
    reinterpret_cast<uint64_t*>(y)[0] = EX_Y[0];
    reinterpret_cast<uint64_t*>(y)[1] = EX_Y[1];
    reinterpret_cast<uint64_t*>(y)[2] = EX_Y[2];
    reinterpret_cast<uint64_t*>(z)[0] = 0;
    reinterpret_cast<uint64_t*>(z)[1] = 0;
    reinterpret_cast<uint64_t*>(z)[2] = 0;

    g_dbgDump = dump;
    ICPU_RUN_KF(mul_ascend, 1, x, y, z, ws, tl);

    printf("INPUT x = %016llx %016llx %016llx\n", (unsigned long long)EX_X[0], (unsigned long long)EX_X[1],
           (unsigned long long)EX_X[2]);
    printf("INPUT y = %016llx %016llx %016llx\n", (unsigned long long)EX_Y[0], (unsigned long long)EX_Y[1],
           (unsigned long long)EX_Y[2]);
    for (int lane = 0; lane < 3; ++lane) {
        const uint64_t prod = EX_X[lane] * EX_Y[lane];
        printf("REF z[%d] = %016llx  (lo32=%08llx hi32=%08llx)\n", lane, (unsigned long long)prod,
               (unsigned long long)(prod & 0xFFFFFFFFULL), (unsigned long long)(prod >> 32));
    }
    printf("OUT z = %016llx %016llx %016llx\n", (unsigned long long)reinterpret_cast<uint64_t*>(z)[0],
           (unsigned long long)reinterpret_cast<uint64_t*>(z)[1],
           (unsigned long long)reinterpret_cast<uint64_t*>(z)[2]);

    for (int slot = 0; slot < 22; ++slot) {
        printf("SLOT %02d | %-22s |", slot, SLOT_NAME[slot] == nullptr ? "-" : SLOT_NAME[slot]);
        for (int k = 0; k < 4; ++k) { printf(" %08x", dump[slot * 8 + k]); }
        printf("\n");
    }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(z);
    AscendC::GmFree(ws);
    AscendC::GmFree(tl);
    AscendC::GmFree(dump);
    return 0;
}
