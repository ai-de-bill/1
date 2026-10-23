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
 * \file reduce_ascend.cpp
 * \brief Grouped sum of an int64 matrix (int64 arithmetic mod 2^64) for Ascend 910B (dav_c220).
 *        VECTOR ONLY: every operation on data is a vector or an MTE (DMA) instruction.
 *
 *   x  : T x M, column-major (column j is the contiguous block [j*T, (j+1)*T) ), int64
 *   idx: T int32 - idx[k] is the row of x that the k-th element of the grouped order refers to
 *   ps : G+1 int32 - group g owns k in [ps[g], ps[g+1]); the last group always ends at T
 *   y  : G x M, column-major:   y[g, j] = sum_{k in group g} x[idx[k], j]     (mod 2^64)
 *
 * The vector unit of 910B has no 64-bit integer arithmetic, so an exact 64-bit sum is assembled from four
 * 16-bit chunk sums.  With v = hh*2^48 + hl*2^32 + lh*2^16 + ll (the four 16-bit chunks of the int64):
 *
 *     sum(v) = (sum ll) + (sum lh) << 16 + (sum hl) << 32 + (sum hh) << 48
 *
 * Data path (no GetValue/SetValue on tensors, no per-element scalar work):
 *   - the whole column is copied into UB with DataCopyPad (MTE2);
 *   - per group and per tile of kTileK values of k: DataCopyPad(idx) -> Muls -> Gather of the low and the
 *     high 32-bit word -> And/ShiftRight into four 16-bit chunk vectors -> Add, accumulated LANE-WISE
 *     into four accumulator vectors (no cross-lane reduction inside the loop);
 *   - after the last tile of a group each accumulator is reduced to lane 0 with an Add tree, the four
 *     lane-0 values are broadcast with Gather (all-zero offsets), combined into the low/high 32-bit word
 *     of the int64 by Shift/Add, and the two words are packed into [lo, hi] and stored with DataCopyPad
 *     (MTE3) - 8 bytes per (group, column).
 *
 * Exactness: a lane accumulates at most ceil(T / kTileK) chunk values of <= 0xFFFF, so with T <= 4096
 * a lane stays below 2^19 and the 512-lane tree total stays below 2^28 - no int32 overflow anywhere.
 *
 * Scalar instructions that remain are pure loop control (the two group boundaries read from prefix_sum
 * and the T/M/G values from the tiling buffer) - no data element is ever touched by the scalar unit.
 *
 * Grid: blockDim = 16 (context->SetBlockDim(16)); block b handles columns j = b, b+16, b+32, ...
 * The whole column must fit into UB, hence T <= kColBufElems (checked by TilingFunc).
 * ---------------------------------------------------------------------------------------------------------- */
#include "kernel_operator.h"

using namespace AscendC;

// Tiling layout.  There is no op_kernel/*_tiling.h: the struct lives here, the kernel entry reads the very
// same numbers out of the tiling GM pointer (exactly like x/idx/ps/y) and TilingFunc
// (op_host/reduce_ascend.cpp) writes them.  REGISTER_TILING_DEFAULT only tells the op build which type the
// tiling has, so the build does not have to load the host tiling library (liboptiling.so requires
// libascend_hal.so, i.e. a driver, which a build machine does not have).
struct ReduceAscendTilingData {
    uint32_t t;
    uint32_t m;
    uint32_t g;
};
REGISTER_TILING_DEFAULT(ReduceAscendTilingData);

namespace {
constexpr int32_t kTileK = 512;          // k values per Gather tile (power of two: the Add tree needs it)
constexpr int32_t kColBufElems = 4096;   // max column length kept in UB (4096 * 8 B = 32 KB)
constexpr int32_t kEvtMte2V = 0;         // MTE2 -> V
constexpr int32_t kEvtVMte3 = 2;         // V    -> MTE3
constexpr int32_t kEvtMte3V = 3;         // MTE3 -> V
constexpr int32_t kOutBytes = static_cast<int32_t>(sizeof(int64_t));   // [lo, hi] of one result

class ReduceAscendKernelPass {
public:
    __aicore__ inline ReduceAscendKernelPass() {}

    __aicore__ inline void Init(__gm__ const int64_t* x, __gm__ const int32_t* idx, __gm__ const int32_t* ps,
                                __gm__ int64_t* y, int64_t t, int64_t m, int64_t g)
    {
        t_ = t;
        m_ = m;
        g_ = g;
        // GlobalTensor only reads x/idx/ps, but SetGlobalBuffer wants a non-const pointer (same as in mul).
        gX_.SetGlobalBuffer((__gm__ int64_t*)x);
        gIdx_.SetGlobalBuffer((__gm__ int32_t*)idx);
        gPs_.SetGlobalBuffer((__gm__ int32_t*)ps);
        gY_.SetGlobalBuffer(y);
        pipe_.InitBuffer(colBuf_, kColBufElems * static_cast<int32_t>(sizeof(int64_t)));
        pipe_.InitBuffer(idxBuf_, kTileK * static_cast<int32_t>(sizeof(int32_t)));
        pipe_.InitBuffer(offBuf_, kTileK * static_cast<int32_t>(sizeof(uint32_t)));
        pipe_.InitBuffer(loBuf_, kTileK * static_cast<int32_t>(sizeof(int32_t)));
        pipe_.InitBuffer(hiBuf_, kTileK * static_cast<int32_t>(sizeof(int32_t)));
        pipe_.InitBuffer(chunkBuf_, 4 * kTileK * static_cast<int32_t>(sizeof(int32_t)));
        pipe_.InitBuffer(maskBuf_, kTileK * static_cast<int32_t>(sizeof(uint32_t)));
        pipe_.InitBuffer(tmpBuf_, kTileK * static_cast<int32_t>(sizeof(int32_t)));
        pipe_.InitBuffer(zeroOffBuf_, kTileK * static_cast<int32_t>(sizeof(uint32_t)));
        pipe_.InitBuffer(tblBuf_, 128);
        pipe_.InitBuffer(outBuf_, 32);
        pipe_.InitBuffer(maskEvenBuf_, 32);
        pipe_.InitBuffer(maskOddBuf_, 32);
    }

    __aicore__ inline void Process()
    {
        if (t_ <= 0 || m_ <= 0 || g_ <= 0 || t_ > kColBufElems) {
            return;
        }
        BuildTables();
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        for (int64_t j = blockIdx; j < m_; j += blockNum) {
            ComputeColumnVector(j);
        }
    }

private:
    // Group g covers k in [GroupStart(g), GroupEnd(g)).  prefix_sum has G+1 entries, so the end of every
    // group is ps[g+1] - except the last group, which always ends at T (with canonical prefix sums that is
    // exactly ps[G]).
    __aicore__ inline int32_t GroupStart(int64_t g) const
    {
        return gPs_.GetValue(g);
    }

    __aicore__ inline int32_t GroupEnd(int64_t g) const
    {
        return (g + 1 < g_) ? gPs_.GetValue(g + 1) : static_cast<int32_t>(t_);
    }

    // Constant tables, built once per kernel.  Only constants are written through SetValue - no data
    // element ever passes through the scalar unit.
    __aicore__ inline void BuildTables()
    {
        // 0x0000FFFF in every 32-bit lane; its uint16 view is [0xFFFF, 0x0000, ...], so the 16-bit And
        // (vand) keeps the low half of each 32-bit word and clears the high half.
        Duplicate(maskBuf_.Get<int32_t>(), static_cast<int32_t>(0x0000FFFF), kTileK);
        // all-zero byte offsets: Gather with this table broadcasts lane 0 of the source to every lane
        Duplicate(zeroOffBuf_.Get<uint32_t>(), static_cast<uint32_t>(0), kTileK);
        // Byte offsets for the tail of the Add tree (three 8-entry, 32-byte tables at offsets 0/32/64).
        // Every table has 8 entries so the Gather/Add pairs stay 32-byte aligned; the repeated entries only
        // pollute lanes that the following step never reads again.
        LocalTensor<uint32_t> tbl = tblBuf_.Get<uint32_t>();
        for (int32_t i = 0; i < 8; ++i) {
            tbl.SetValue(i, static_cast<uint32_t>(16 + 4 * (i % 4)));       // v[4..8)
            tbl.SetValue(8 + i, static_cast<uint32_t>(8 + 4 * (i % 2)));    // v[2..4)
            tbl.SetValue(16 + i, static_cast<uint32_t>(4));                 // v[1]
        }
        // packing [lo, hi]: lane 0 keeps the low word only, lane 1 the high word only
        LocalTensor<int32_t> evenMask = maskEvenBuf_.Get<int32_t>();
        evenMask.SetValue(0, -1);
        evenMask.SetValue(1, 0);
        LocalTensor<int32_t> oddMask = maskOddBuf_.Get<int32_t>();
        oddMask.SetValue(0, 0);
        oddMask.SetValue(1, -1);
        PipeBarrier<PIPE_ALL>();
    }

    // One column through the vector unit: the column is copied into UB once, then for every group the rows
    // are gathered in tiles of kTileK elements, split into 16-bit chunks and reduced.  The int64 sum is the
    // weighted sum of the four chunk totals (see the file header).
    __aicore__ inline void ComputeColumnVector(int64_t j)
    {
        LocalTensor<int64_t> col = colBuf_.Get<int64_t>();
        DataCopyExtParams colParams{1, static_cast<uint32_t>(t_ * static_cast<int64_t>(sizeof(int64_t))), 0, 0, 0};
        DataCopyPadExtParams<int64_t> colPad{false, 0, 0, 0};
        DataCopyPad(col, gX_[j * t_], colParams, colPad);
        SetFlag<HardEvent::MTE2_V>(kEvtMte2V);
        WaitFlag<HardEvent::MTE2_V>(kEvtMte2V);

        const LocalTensor<int32_t> col32 = col.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> idxLoc = idxBuf_.Get<int32_t>();
        LocalTensor<int32_t> offs = offBuf_.Get<int32_t>();
        LocalTensor<uint32_t> offsU = offs.ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> lo = loBuf_.Get<int32_t>();
        LocalTensor<int32_t> hi = hiBuf_.Get<int32_t>();
        LocalTensor<uint32_t> loU = lo.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> hiU = hi.ReinterpretCast<uint32_t>();
        LocalTensor<uint16_t> lo16 = lo.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> hi16 = hi.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> mask16 = maskBuf_.Get<uint16_t>();
        LocalTensor<int32_t> chunks = chunkBuf_.Get<int32_t>();
        LocalTensor<int32_t> c0 = chunks[0];
        LocalTensor<int32_t> c1 = chunks[kTileK];
        LocalTensor<int32_t> c2 = chunks[2 * kTileK];
        LocalTensor<int32_t> c3 = chunks[3 * kTileK];
        LocalTensor<int32_t> tmp = tmpBuf_.Get<int32_t>();
        LocalTensor<uint32_t> tmpU = tmp.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> zeroOff = zeroOffBuf_.Get<uint32_t>();
        LocalTensor<int32_t> out = outBuf_.Get<int32_t>();

        for (int64_t g = 0; g < g_; ++g) {
            const int32_t k0 = GroupStart(g);
            const int32_t k1 = GroupEnd(g);
            // Four lane-wise accumulators (one per 16-bit chunk); nothing is reduced inside the tile loop.
            Duplicate(c0, 0, kTileK);
            Duplicate(c1, 0, kTileK);
            Duplicate(c2, 0, kTileK);
            Duplicate(c3, 0, kTileK);
            for (int32_t base = k0; base < k1; base += kTileK) {
                const int32_t cnt = ((k1 - base) < kTileK) ? (k1 - base) : kTileK;
                // idx tile -> byte offsets of the int64 elements inside the column
                DataCopyExtParams idxParams{1, static_cast<uint32_t>(cnt * static_cast<int32_t>(sizeof(int32_t))),
                                            0, 0, 0};
                DataCopyPadExtParams<int32_t> idxPad{false, 0, 0, 0};
                DataCopyPad(idxLoc, gIdx_[base], idxParams, idxPad);
                SetFlag<HardEvent::MTE2_V>(kEvtMte2V);
                WaitFlag<HardEvent::MTE2_V>(kEvtMte2V);
                Muls(offs, idxLoc, static_cast<int32_t>(sizeof(int64_t)), cnt);
                PipeBarrier<PIPE_V>();
                if (cnt < kTileK) {
                    // A partial tile must not contribute the lanes past cnt and those lanes may still hold
                    // the previous tile, so the whole vector is zeroed first (at offset 0, which is 32B
                    // aligned); Gather then fills [0, cnt) and leaves [cnt, kTileK) at zero.
                    Duplicate(lo, 0, kTileK);
                    Duplicate(hi, 0, kTileK);
                    PipeBarrier<PIPE_V>();
                }
                // low word of x[idx[k], j] (base 0) and high word (base sizeof(int32))
                Gather(lo, col32, offsU, 0, static_cast<uint32_t>(cnt));
                Gather(hi, col32, offsU, static_cast<uint32_t>(sizeof(int32_t)), static_cast<uint32_t>(cnt));
                PipeBarrier<PIPE_V>();
                // the four 16-bit chunks of the gathered values, added lane-wise
                And(tmp.ReinterpretCast<uint16_t>(), lo16, mask16, 2 * kTileK);
                Add(c0, c0, tmp, kTileK);
                ShiftRight(tmpU, loU, static_cast<uint32_t>(16), kTileK);
                Add(c1, c1, tmp, kTileK);
                And(tmp.ReinterpretCast<uint16_t>(), hi16, mask16, 2 * kTileK);
                Add(c2, c2, tmp, kTileK);
                ShiftRight(tmpU, hiU, static_cast<uint32_t>(16), kTileK);
                Add(c3, c3, tmp, kTileK);
                PipeBarrier<PIPE_V>();
            }
            StoreGroupResult(g, j, c0, c1, c2, c3, zeroOff, out);
        }
        PipeBarrier<PIPE_ALL>();
    }

    // In-place reduction of the kTileK int32 lanes of v: the total ends up in lane 0.  The tree uses only
    // 32-byte aligned views down to 8 lanes; the last three steps go through Gather with constant offset
    // tables, because v[4] / v[2] / v[1] would violate the 32-byte tensor alignment rule.
    __aicore__ inline void ReduceToLane0(const LocalTensor<int32_t>& v)
    {
        for (int32_t len = kTileK; len > 8; len >>= 1) {
            Add(v, v, v[len >> 1], len >> 1);
        }
        LocalTensor<int32_t> tmp = tmpBuf_.Get<int32_t>();
        LocalTensor<uint32_t> tbl = tblBuf_.Get<uint32_t>();
        Gather(tmp, v, tbl, 0, 8);
        Add(v, v, tmp, 8);              // 8 -> 4 lanes
        Gather(tmp, v, tbl[8], 0, 8);   // second table, byte offset 32
        Add(v, v, tmp, 8);              // 4 -> 2 lanes
        Gather(tmp, v, tbl[16], 0, 8);  // third table, byte offset 64
        Add(v, v, tmp, 8);              // 2 -> 1 lane
    }

    // y[g, j]: the four chunk totals (each broadcast from lane 0) are combined into the low and the high
    // 32-bit word of the int64, packed into [lo, hi] and stored with one 8-byte DataCopyPad.
    __aicore__ inline void StoreGroupResult(int64_t g, int64_t j, const LocalTensor<int32_t>& c0,
                                            const LocalTensor<int32_t>& c1, const LocalTensor<int32_t>& c2,
                                            const LocalTensor<int32_t>& c3, const LocalTensor<uint32_t>& zeroOff,
                                            const LocalTensor<int32_t>& out)
    {
        LocalTensor<int32_t> lo = loBuf_.Get<int32_t>();
        LocalTensor<int32_t> hi = hiBuf_.Get<int32_t>();
        LocalTensor<int32_t> tmp = tmpBuf_.Get<int32_t>();
        // Shifts exist only for unsigned 32-bit lanes (that is also what keeps them logical).
        LocalTensor<uint32_t> loU = lo.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> hiU = hi.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> tmpU = tmp.ReinterpretCast<uint32_t>();
        ReduceToLane0(c0);
        ReduceToLane0(c1);
        ReduceToLane0(c2);
        ReduceToLane0(c3);
        // broadcast the chunk totals (lane 0 of each accumulator) to every lane
        Gather(tmp, c0, zeroOff, 0, kTileK);
        Gather(lo, c1, zeroOff, 0, kTileK);
        // carry into the high word: ((c0 >> 16) + c1) >> 16
        ShiftRight(hiU, tmpU, static_cast<uint32_t>(16), kTileK);
        Add(hi, hi, lo, kTileK);
        ShiftRight(hiU, hiU, static_cast<uint32_t>(16), kTileK);
        // low 32 bits: c0 + (c1 << 16)
        ShiftLeft(loU, loU, static_cast<uint32_t>(16), kTileK);
        Add(lo, lo, tmp, kTileK);
        // high 32 bits: c2 + (c3 << 16) + carry
        Gather(tmp, c2, zeroOff, 0, kTileK);
        Add(hi, hi, tmp, kTileK);
        Gather(tmp, c3, zeroOff, 0, kTileK);
        ShiftLeft(tmpU, tmpU, static_cast<uint32_t>(16), kTileK);
        Add(hi, hi, tmp, kTileK);
        PipeBarrier<PIPE_V>();
        // pack [lo, hi]: lane 0 keeps the low word, lane 1 the high word (16-bit And on the uint16 view)
        And(out.ReinterpretCast<uint16_t>(), lo.ReinterpretCast<uint16_t>(), maskEvenBuf_.Get<uint16_t>(), 16);
        And(tmp.ReinterpretCast<uint16_t>(), hi.ReinterpretCast<uint16_t>(), maskOddBuf_.Get<uint16_t>(), 16);
        Add(out, out, tmp, 8);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(kEvtVMte3);
        WaitFlag<HardEvent::V_MTE3>(kEvtVMte3);
        DataCopyExtParams oParams{1, static_cast<uint32_t>(kOutBytes), 0, 0, 0};
        DataCopyPad(gY_[g + j * g_], out.ReinterpretCast<int64_t>(), oParams);
        SetFlag<HardEvent::MTE3_V>(kEvtMte3V);
        WaitFlag<HardEvent::MTE3_V>(kEvtMte3V);
    }

    TPipe pipe_;
    GlobalTensor<int64_t> gX_;
    GlobalTensor<int32_t> gIdx_;
    GlobalTensor<int32_t> gPs_;
    GlobalTensor<int64_t> gY_;
    TBuf<> colBuf_;
    TBuf<> idxBuf_;
    TBuf<> offBuf_;
    TBuf<> loBuf_;
    TBuf<> hiBuf_;
    TBuf<> chunkBuf_;
    TBuf<> maskBuf_;
    TBuf<> tmpBuf_;
    TBuf<> zeroOffBuf_;
    TBuf<> tblBuf_;
    TBuf<> outBuf_;
    TBuf<> maskEvenBuf_;
    TBuf<> maskOddBuf_;
    int64_t t_ = 0;   // rows of x / len(idx)
    int64_t m_ = 0;   // columns
    int64_t g_ = 0;   // groups
};
} // namespace

extern "C" __global__ __aicore__ void reduce_ascend(GM_ADDR x, GM_ADDR idx, GM_ADDR ps, GM_ADDR y,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    // The tiling arrives as a plain GM pointer, exactly like x/idx/ps/y: no tiling struct, no
    // REGISTER_TILING_DEFAULT / GET_TILING_DATA, so op_kernel/*_tiling.h does not exist.  Layout, written
    // by TilingFunc (see op_host/reduce_ascend.cpp):
    //     tiling[0] = T (rows of x == len(idx)),  tiling[1] = M (columns),  tiling[2] = G (groups)
    GlobalTensor<uint32_t> tilingGm;
    tilingGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tiling));
    const int64_t t = static_cast<int64_t>(tilingGm.GetValue(0));
    const int64_t m = static_cast<int64_t>(tilingGm.GetValue(1));
    const int64_t g = static_cast<int64_t>(tilingGm.GetValue(2));

    ReduceAscendKernelPass op;
    op.Init(reinterpret_cast<__gm__ const int64_t*>(x), reinterpret_cast<__gm__ const int32_t*>(idx),
            reinterpret_cast<__gm__ const int32_t*>(ps), reinterpret_cast<__gm__ int64_t*>(y), t, m, g);
    op.Process();
}
