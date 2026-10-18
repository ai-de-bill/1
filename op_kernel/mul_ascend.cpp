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
 * \file mul_ascend.cpp
 * \brief Element-wise int64 * int64 -> int64 for Ascend 910B (dav_c220).
 *
 * The 910B vector unit has no 64-bit integer multiply, so the product is assembled from 32-bit
 * vector primitives. With  a = aLo + (aHi << 32),  b = bLo + (bHi << 32)  (two's complement halves):
 *
 *     a * b = (aLo * bLo) + ((aLo * bHi + aHi * bLo) << 32)      (mod 2^64)
 *
 * The low 32 bits of a 32x32 product is a plain 32-bit multiply and only aLo * bLo needs its high
 * 32 bits, so one tile of TILE_ELEMS elements costs
 *
 *     rLo = Mul(aLo, bLo)                                  1 vector op
 *     p1  = Mul(aLo, bHi)                                  1 vector op (only the low half survives)
 *     p2  = Mul(aHi, bLo)                                  1 vector op (only the low half survives)
 *     rHi = MulHi(aLo, bLo) + p1 + p2                      22 vector ops + 2 adds
 *
 * MulHi(x, y) = high32(x * y) is evaluated from the 16-bit split x = x1*2^16 + x0, y = y1*2^16 + y0
 * without any bitwise carry trick:
 *
 *     x * y = A + T*2^16 + W*2^32,   A = x0*y0, T = x0*y1 + x1*y0, W = x1*y1
 *     U = x0*y1 = U1*2^16 + U0,  V = x1*y0 = V1*2^16 + V0
 *     s0 = U0 + V0  ->  T_lo = s0 & 0xFFFF, c0 = s0 >> 16          (s0 < 2^17, exact)
 *     T_hi = U1 + V1 + c0                                          (equals T >> 16 exactly)
 *     high32 = W + T_hi + (((A >> 16) + T_lo) >> 16)
 *
 * Other 910B specifics that shape the code:
 *   - Mul/Add/Sub exist only for half/float/int16/int32 (no unsigned overload), hence the int32
 *     view for arithmetic: results modulo 2^32 do not depend on the signedness;
 *   - ShiftLeft/ShiftRight with a *scalar* amount on uint32 keep the shift logical (the overload
 *     taking a shift tensor exists only on other SoCs);
 *   - And/Or/Not are 16-bit instructions (vand/vor/vnot); they are applied to the uint16 view of
 *     the buffers with count = 2 * elementCount, which yields an exact 32-bit wide bitwise op.
 *     The 0xFFFF mask buffer is a Duplicate of 0x0000FFFF, so its 16-bit view is the needed
 *     [0xFFFF, 0x0000, ...] pattern;
 *   - Gather takes byte offsets: with x and y placed back to back in one buffer, a single Gather
 *     per half produces [x_lo | y_lo] and [x_hi | y_hi];
 *   - synchronization is explicit (no TQue): the kernel runs a hand-written double-buffered
 *     pipeline over PIPE_MTE2 (GM -> UB), PIPE_V (compute) and PIPE_MTE3 (UB -> GM):
 *
 *       MTE2 (load tile i+1) ----+
 *       V    (compute tile i) ---+--> overlap of the three pipelines
 *       MTE3 (store tile i-1) ---+
 *
 *     Four event pairs are used, one per ping-pong buffer:
 *       MTE2_V : the load of buffer b is complete          -> V may read it
 *       V_MTE2 : V finished reading buffer b               -> MTE2 may overwrite it
 *       V_MTE3 : V finished writing the output buffer b    -> MTE3 may read it
 *       MTE3_V : MTE3 finished reading the output buffer b -> V may overwrite it
 *     PIPE_MTE1 is not involved at all: this is a vector-only kernel, there is no L1/L0A/L0B
 *     traffic (GM -> UB copies are done by MTE2 directly).  PIPE_S is touched only inside the
 *     toolkit (CreateVecIndex uses SetValue plus its own S_V/V_S syncs while the tables are built).
 */

#include "kernel_operator.h"
#include "mul_ascend_tiling.h"

using namespace AscendC;

namespace {
// int64 elements handled by one vector tile.
constexpr int32_t TILE_ELEMS = 512;
// 32-bit lanes covered by one tile (one int64 element == two uint32 halves).
constexpr int32_t TILE_WORDS = TILE_ELEMS * 2;
// Depth of the ping-pong double buffer (input and output buffers are swapped each tile).
constexpr int32_t BUF_DEPTH = 2;
// Shift used to split/recombine the 16-bit halves of a 32-bit lane.
constexpr uint32_t SH16 = 16u;
// Size of one int64 element in bytes, and of one 32-bit half inside it.
constexpr int32_t ELEM_BYTES = 8;
constexpr int32_t WORD_BYTES = 4;
// Size of one input buffer (x and y back to back) and of one output buffer.
constexpr int32_t IN_BUF_BYTES = TILE_WORDS * ELEM_BYTES;
constexpr int32_t OUT_BUF_BYTES = TILE_ELEMS * ELEM_BYTES;
// Event ids of the manual pipeline: 0 for the first ping-pong buffer, 1 for the second.  The flag
// space belongs to a (source pipe, destination pipe) pair, so the same two ids are reused for all
// four handshakes (MTE2_V, V_MTE2, V_MTE3, MTE3_V).  Fixed ids are used instead of
// TPipe::FetchEventID() on purpose: the CPU debug model does not model the per-event-type id pool
// and returns 0 for every call, so the model's checker ("an event id cannot be set twice
// consecutively") would fire on the priming of the pipeline.
constexpr int32_t EVT_ID[BUF_DEPTH] = {0, 1};
// Optional intra-pipe (PIPE_V -> PIPE_V) barriers, disabled by default.
//
// They are NOT required for the instruction families used by this kernel: plain element-wise vector
// ops (Mul/Add/And/ShiftLeft/ShiftRight/Duplicate) and Gather are executed in order inside PIPE_V,
// so a RAW dependency between two of them (e.g. Add(s1, u, v) followed by Add(s1, s1, c0)) is
// resolved by the hardware itself.  The CANN toolkit follows exactly the same rule for dav_c220:
// its own implementations put PipeBarrier<PIPE_V>() only around the families whose result appears
// through a special path - WholeReduce/RepeatReduce (kernel_operator_vec_reduce_impl.h), Transpose,
// Compare/Select and BilinearInterpolation - while vec_binary / vec_binary_scalar / vec_unary /
// vec_duplicate / vec_gather / vec_scatter / vec_brcb contain no intra-pipe barrier at all.
// Note also that SetFlag<HardEvent::V_V>() itself degenerates into PipeBarrier<PIPE_V>()
// (impl/basic_api/kernel_event.h), i.e. there is no "event" ordering inside one pipe.
//
// Define MUL_ASCEND_V_BARRIERS=1 to insert a few barriers anyway (diagnostics / A-B measurement).
#ifndef MUL_ASCEND_V_BARRIERS
#define MUL_ASCEND_V_BARRIERS 0
#endif

class MulAscendKernelPass {
public:
    __aicore__ inline MulAscendKernelPass() {}

    __aicore__ inline void Init(__gm__ const int64_t* x, __gm__ const int64_t* y, __gm__ int64_t* z, int64_t n)
    {
        const int64_t blockIdx = GetBlockIdx();
        const int64_t blockNum = GetBlockNum();
        const int64_t elemsPerBlock = (n + blockNum - 1) / blockNum;

        start_ = blockIdx * elemsPerBlock;
        end_ = start_ + elemsPerBlock;
        if (end_ > n) {
            end_ = n;
        }
        valid_ = (start_ < end_);
        if (!valid_) {
            return;
        }

        const uint64_t localCount = static_cast<uint64_t>(end_ - start_);
        gX_.SetGlobalBuffer((__gm__ int64_t*)x + start_, localCount);
        gY_.SetGlobalBuffer((__gm__ int64_t*)y + start_, localCount);
        gZ_.SetGlobalBuffer((__gm__ int64_t*)z + start_, localCount);

        // Ping-pong buffers of the manual pipeline: one input buffer holds x in [0, TILE_ELEMS) and
        // y in [TILE_ELEMS, TILE_WORDS) (so a single Gather builds both halves of both operands),
        // the paired output buffer holds one result tile.
        for (int32_t b = 0; b < BUF_DEPTH; ++b) {
            pipe_.InitBuffer(inBuf_[b], IN_BUF_BYTES);
            pipe_.InitBuffer(outBuf_[b], OUT_BUF_BYTES);
        }

        // Gather index vectors and the 16-bit mask (all offsets are expressed in bytes).
        pipe_.InitBuffer(tLoIdx_, TILE_WORDS * WORD_BYTES);
        pipe_.InitBuffer(tInterIdx_, TILE_WORDS * WORD_BYTES);
        pipe_.InitBuffer(tMask_, TILE_ELEMS * WORD_BYTES);

        // Split operands: tLoBuf_ = [aLo | bLo], tHiBuf_ = [aHi | bHi].
        pipe_.InitBuffer(tLoBuf_, TILE_WORDS * WORD_BYTES);
        pipe_.InitBuffer(tHiBuf_, TILE_WORDS * WORD_BYTES);
        // Result: tPack_[0, TILE_ELEMS) = rLo, tPack_[TILE_ELEMS, TILE_WORDS) = rHi.
        pipe_.InitBuffer(tPack_, TILE_WORDS * WORD_BYTES);
        pipe_.InitBuffer(tP1Lo_, TILE_ELEMS * WORD_BYTES);

        // Scratch of MulHi.
        pipe_.InitBuffer(tMx0_, TILE_ELEMS * WORD_BYTES);
        pipe_.InitBuffer(tMy0_, TILE_ELEMS * WORD_BYTES);
        pipe_.InitBuffer(tMx1_, TILE_ELEMS * WORD_BYTES);
        pipe_.InitBuffer(tMy1_, TILE_ELEMS * WORD_BYTES);
        pipe_.InitBuffer(tMu_, TILE_ELEMS * WORD_BYTES);
        pipe_.InitBuffer(tMv_, TILE_ELEMS * WORD_BYTES);
        pipe_.InitBuffer(tMw_, TILE_ELEMS * WORD_BYTES);
        pipe_.InitBuffer(tMa_, TILE_ELEMS * WORD_BYTES);

        BuildTables();
    }

    __aicore__ inline void Process()
    {
        if (!valid_) {
            return;
        }

        const int64_t total = end_ - start_;
        const int32_t tiles = static_cast<int32_t>((total + TILE_ELEMS - 1) / TILE_ELEMS);

        // Prime the pipeline: both input buffers are free for MTE2 and both output buffers are free
        // for V, so that the first loads/stores do not wait for a flag that nobody has set yet.
        for (int32_t b = 0; b < BUF_DEPTH; ++b) {
            SetFlag<HardEvent::V_MTE2>(EVT_ID[b]);
            SetFlag<HardEvent::MTE3_V>(EVT_ID[b]);
        }

        // Prologue: issue the load of the very first tile.
        CopyIn(TileOffset(0), TileCount(0, total), 0);

        for (int32_t i = 0; i < tiles; ++i) {
            const int32_t b = i % BUF_DEPTH;

            // Issue the load of the next tile into the other buffer: MTE2 then runs in parallel with
            // the vector work below (that load waits for V_MTE2 of its buffer, i.e. for the compute
            // of tile i-1 which used it).
            if (i + 1 < tiles) {
                CopyIn(TileOffset(i + 1), TileCount(i + 1, total), (i + 1) % BUF_DEPTH);
            }

            // The input data of tile i is in UB ...
            WaitFlag<HardEvent::MTE2_V>(EVT_ID[b]);
            // ... and the output buffer is free (its store finished, or the primed flag is consumed).
            WaitFlag<HardEvent::MTE3_V>(EVT_ID[b]);

            Compute(inBuf_[b].Get<int64_t>(), outBuf_[b].Get<int64_t>(), TileCount(i, total));

            // V is done with both buffers of this tile: the input buffer may be reloaded, and the
            // store of the output buffer may start (CopyOut performs the V_MTE3 handshake).
            SetFlag<HardEvent::V_MTE2>(EVT_ID[b]);
            CopyOut(TileOffset(i), TileCount(i, total), b);
        }

        // Drain the flags that stay pending for the last tiles, so that the kernel does not leave
        // half-open events behind (this also waits for the very last store to complete).
        for (int32_t b = 0; b < BUF_DEPTH; ++b) {
            WaitFlag<HardEvent::V_MTE2>(EVT_ID[b]);
            WaitFlag<HardEvent::MTE3_V>(EVT_ID[b]);
        }
    }

    // Offset of tile i inside the range owned by this block, and the number of elements in it.
    __aicore__ inline int64_t TileOffset(int32_t i) const
    {
        return static_cast<int64_t>(i) * TILE_ELEMS;
    }

    __aicore__ inline int32_t TileCount(int32_t i, int64_t total) const
    {
        const int64_t left = total - TileOffset(i);
        return (left > TILE_ELEMS) ? TILE_ELEMS : static_cast<int32_t>(left);
    }

private:
    // Builds, once per kernel launch, the tables used by the tile loop:
    //   tLoIdx_[j]    = 8*j        -> byte offset of the low 32 bits of the j-th half; the high
    //                                 halves use the same vector with srcBaseOffset = 4
    //     (x and y sit back to back, so walking 8*j covers x's halves and then y's halves)
    //   tInterIdx_[j] = 4*(j/2) + (j%2)*4*TILE_ELEMS
    //                              -> re-interleaves [rLo | rHi] into the int64 layout
    //   tMask_        = 0x0000FFFF per 32-bit lane, i.e. [0xFFFF, 0x0000, ...] in the uint16 view
    __aicore__ inline void BuildTables()
    {
        // Only one byte-offset vector is needed: the high halves are the same offsets plus one
        // 32-bit word, which Gather takes from its srcBaseOffset argument.
        auto loIdx = tLoIdx_.Get<int32_t>();
        CreateVecIndex(loIdx, 0, static_cast<uint32_t>(TILE_WORDS));
        Muls(loIdx, loIdx, static_cast<int32_t>(ELEM_BYTES), TILE_WORDS); // 8 * j

        // tPack_ and tLoBuf_ are unused until the first tile is processed, so they serve as the two
        // scratch vectors of the interleave index instead of two dedicated buffers.
        auto interIdx = tInterIdx_.Get<int32_t>();
        auto tmp0 = tPack_.Get<int32_t>();
        auto tmp1 = tLoBuf_.Get<int32_t>();
        CreateVecIndex(interIdx, 0, static_cast<uint32_t>(TILE_WORDS));
        ShiftRight(tmp0, interIdx, 1, TILE_WORDS);              // j / 2
        ShiftLeft(tmp1, tmp0, 1, TILE_WORDS);                   // 2 * (j / 2)
        Sub(tmp1, interIdx, tmp1, TILE_WORDS);                  // j % 2
        Muls(tmp1, tmp1, TILE_ELEMS * WORD_BYTES, TILE_WORDS);  // (j % 2) * byte size of rLo
        Muls(tmp0, tmp0, WORD_BYTES, TILE_WORDS);               // 4 * (j / 2)
        Add(interIdx, tmp0, tmp1, TILE_WORDS);

        Duplicate(tMask_.Get<int32_t>(), static_cast<int32_t>(0x0000FFFF), TILE_ELEMS);
    }

    // GM -> UB (PIPE_MTE2).  The buffer may only be overwritten after V is done with its previous
    // content, and V may only read it once this copy is complete.
    __aicore__ inline void CopyIn(int64_t off, int32_t cnt, int32_t buf)
    {
        auto inLocal = inBuf_[buf].Get<int64_t>();

        WaitFlag<HardEvent::V_MTE2>(EVT_ID[buf]);

        // DataCopyPad is required because cnt * sizeof(int64_t) is not necessarily a multiple of 32B.
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(cnt * ELEM_BYTES), 0, 0, 0};
        DataCopyPadExtParams<int64_t> padParams{false, 0, 0, 0};
        DataCopyPad(inLocal[0], gX_[off], copyParams, padParams);
        DataCopyPad(inLocal[TILE_ELEMS], gY_[off], copyParams, padParams);

        SetFlag<HardEvent::MTE2_V>(EVT_ID[buf]);
    }

    // UB -> GM (PIPE_MTE3).  The store must wait for V to finish the tile, and V must not overwrite
    // the buffer before the store has been issued (V_MTE3 handshake), nor before it is finished
    // (MTE3_V, waited in Process before the next Compute that reuses this buffer).
    __aicore__ inline void CopyOut(int64_t off, int32_t cnt, int32_t buf)
    {
        SetFlag<HardEvent::V_MTE3>(EVT_ID[buf]);
        WaitFlag<HardEvent::V_MTE3>(EVT_ID[buf]);

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(cnt * ELEM_BYTES), 0, 0, 0};
        DataCopyPad(gZ_[off], outBuf_[buf].Get<int64_t>(), copyParams);

        SetFlag<HardEvent::MTE3_V>(EVT_ID[buf]);
    }


    __aicore__ inline void Compute(const LocalTensor<int64_t>& in64, const LocalTensor<int64_t>& out64,
                                   int32_t cnt)
    {
        auto in32 = in64.ReinterpretCast<uint32_t>();   // 4 * TILE_ELEMS lanes
        auto out32 = out64.ReinterpretCast<uint32_t>(); // 2 * TILE_ELEMS lanes

        auto loBuf = tLoBuf_.Get<uint32_t>();
        auto hiBuf = tHiBuf_.Get<uint32_t>();

        // Split into 32-bit halves: tLoBuf_ = [aLo | bLo], tHiBuf_ = [aHi | bHi]. The gathers always
        // run over the whole tile; for a short tail tile the extra lanes are simply not used.  The
        // high halves sit at the same offsets plus one 32-bit word, which is what srcBaseOffset
        // supplies - hence a single index vector is enough for both gathers.
        Gather(loBuf, in32, tLoIdx_.Get<uint32_t>(), 0, static_cast<uint32_t>(TILE_WORDS));
        Gather(hiBuf, in32, tLoIdx_.Get<uint32_t>(), WORD_BYTES, static_cast<uint32_t>(TILE_WORDS));
#if MUL_ASCEND_V_BARRIERS
        // insurance only, see MUL_ASCEND_V_BARRIERS: Gather is an ordinary in-order PIPE_V op
        PipeBarrier<PIPE_V>();
#endif

        auto aLo = loBuf;
        auto bLo = loBuf[TILE_ELEMS];
        auto aHi = hiBuf;
        auto bHi = hiBuf[TILE_ELEMS];

        auto pack = tPack_.Get<uint32_t>();
        auto rLo = pack;             // pack[0, TILE_ELEMS)          -> low 32 bits of the result
        auto rHi = pack[TILE_ELEMS]; // pack[TILE_ELEMS, TILE_WORDS) -> high 32 bits of the result
        auto cross = tP1Lo_.Get<uint32_t>(); // scratch for the two cross terms

        // rHi = high32(aLo * bLo); the cross terms are added one after another and each contributes
        // only its low 32 bits (the << 32 shift), so one scratch buffer serves both of them.
        MulHi(aLo, bLo, rHi, cnt);
#if MUL_ASCEND_V_BARRIERS
        // insurance only, see MUL_ASCEND_V_BARRIERS: MulHi is a chain of in-order PIPE_V ops
        PipeBarrier<PIPE_V>();
#endif
        // rLo = low32(aLo * bLo)
        Mul(rLo.ReinterpretCast<int32_t>(), aLo.ReinterpretCast<int32_t>(),
            bLo.ReinterpretCast<int32_t>(), cnt);
        Mul(cross.ReinterpretCast<int32_t>(), aLo.ReinterpretCast<int32_t>(),
            bHi.ReinterpretCast<int32_t>(), cnt);
        Add(rHi.ReinterpretCast<int32_t>(), rHi.ReinterpretCast<int32_t>(),
            cross.ReinterpretCast<int32_t>(), cnt);
        Mul(cross.ReinterpretCast<int32_t>(), aHi.ReinterpretCast<int32_t>(),
            bLo.ReinterpretCast<int32_t>(), cnt);
        Add(rHi.ReinterpretCast<int32_t>(), rHi.ReinterpretCast<int32_t>(),
            cross.ReinterpretCast<int32_t>(), cnt);

        // Re-interleave [rLo | rHi] into the int64 layout of the output tile.  All of the above runs
        // on PIPE_V, so these vector operations are guaranteed to execute in program order.
        Gather(out32, pack, tInterIdx_.Get<uint32_t>(), 0, static_cast<uint32_t>(2 * cnt));
    }

    // hiOut = high32(x * y) for unsigned 32-bit lanes, x = x1*2^16 + x0, y = y1*2^16 + y0.
    // See the file header for the derivation; there is no bitwise carry trick here, T is rebuilt
    // exactly from the 16-bit halves of U = x0*y1 and V = x1*y0.
    //
    // Only eight scratch buffers are used: the four operand halves die as soon as the last product
    // that needs them is computed, so their buffers are taken over by U0/V0/s0/c0 (the products are
    // evaluated in the order A, U, V, W so that x0 dies after U and y0 after V), and U/V are turned
    // into U1/V1 in place.  Peak usage is therefore 4 products + 2 live operand halves + 1 new
    // result = 7 lanes-worth of buffers.
    __aicore__ inline void MulHi(const LocalTensor<uint32_t>& x, const LocalTensor<uint32_t>& y,
                                 const LocalTensor<uint32_t>& hiOut, int32_t cnt)
    {
        const int32_t cnt16 = 2 * cnt;
        auto mask16 = tMask_.Get<uint16_t>();

        auto x0 = tMx0_.Get<uint32_t>(); // x0 -> U0
        auto y0 = tMy0_.Get<uint32_t>(); // y0 -> V0
        auto x1 = tMx1_.Get<uint32_t>(); // x1 -> s0 -> T_lo
        auto y1 = tMy1_.Get<uint32_t>(); // y1 -> c0
        auto a = tMa_.Get<uint32_t>();   // A  -> A_hi -> S -> S_hi
        auto u = tMu_.Get<uint32_t>();   // U  -> U1 -> T_hi
        auto v = tMv_.Get<uint32_t>();   // V  -> V1
        auto w = tMw_.Get<uint32_t>();   // W

        // Low/high 16-bit halves of the operands.
        And(x0.ReinterpretCast<uint16_t>(), x.ReinterpretCast<uint16_t>(), mask16, cnt16);
        And(y0.ReinterpretCast<uint16_t>(), y.ReinterpretCast<uint16_t>(), mask16, cnt16);
        ShiftRight(x1, x, SH16, cnt);
        ShiftRight(y1, y, SH16, cnt);

        // Products, in an order that frees the operand buffers as early as possible.
        Mul(a.ReinterpretCast<int32_t>(), x0.ReinterpretCast<int32_t>(),
            y0.ReinterpretCast<int32_t>(), cnt);   // A = x0 * y0
        Mul(u.ReinterpretCast<int32_t>(), x0.ReinterpretCast<int32_t>(),
            y1.ReinterpretCast<int32_t>(), cnt);   // U = x0 * y1  (x0 is dead after this)
        Mul(v.ReinterpretCast<int32_t>(), x1.ReinterpretCast<int32_t>(),
            y0.ReinterpretCast<int32_t>(), cnt);   // V = x1 * y0  (y0 is dead after this)
        Mul(w.ReinterpretCast<int32_t>(), x1.ReinterpretCast<int32_t>(),
            y1.ReinterpretCast<int32_t>(), cnt);   // W = x1 * y1  (x1/y1 are dead after this)

        // T = U + V rebuilt exactly: s0 = U0 + V0 (< 2^17) gives T_lo = s0 & 0xFFFF and c0 = s0 >> 16,
        // then T_hi = U1 + V1 + c0 (which equals T >> 16 exactly, T_hi < 2^17).
        // U0/V0 land in the dead x0/y0 buffers, s0/c0 in the dead x1/y1 ones.
        And(x0.ReinterpretCast<uint16_t>(), u.ReinterpretCast<uint16_t>(), mask16, cnt16); // U0
        And(y0.ReinterpretCast<uint16_t>(), v.ReinterpretCast<uint16_t>(), mask16, cnt16); // V0
        ShiftRight(u, u, SH16, cnt);                        // U becomes U1
        ShiftRight(v, v, SH16, cnt);                        // V becomes V1
        Add(x1.ReinterpretCast<int32_t>(), x0.ReinterpretCast<int32_t>(),
            y0.ReinterpretCast<int32_t>(), cnt);            // s0 = U0 + V0
        ShiftRight(y1, x1, SH16, cnt);                      // c0 = s0 >> 16
        And(x1.ReinterpretCast<uint16_t>(), x1.ReinterpretCast<uint16_t>(), mask16, cnt16);
        // x1 now holds T_lo
        Add(u.ReinterpretCast<int32_t>(), u.ReinterpretCast<int32_t>(),
            v.ReinterpretCast<int32_t>(), cnt);             // U1 + V1
        Add(u.ReinterpretCast<int32_t>(), u.ReinterpretCast<int32_t>(),
            y1.ReinterpretCast<int32_t>(), cnt);            // u now holds T_hi

        // high32 = W + T_hi + (((A >> 16) + T_lo) >> 16)
        ShiftRight(a, a, SH16, cnt);                        // A becomes A_hi
        Add(a.ReinterpretCast<int32_t>(), a.ReinterpretCast<int32_t>(),
            x1.ReinterpretCast<int32_t>(), cnt);            // A_hi + T_lo
        ShiftRight(a, a, SH16, cnt);                        // becomes the final carry term
        Add(hiOut.ReinterpretCast<int32_t>(), w.ReinterpretCast<int32_t>(),
            u.ReinterpretCast<int32_t>(), cnt);             // W + T_hi
        Add(hiOut.ReinterpretCast<int32_t>(), hiOut.ReinterpretCast<int32_t>(),
            a.ReinterpretCast<int32_t>(), cnt);             // + S_hi
    }

    TPipe pipe_;

    GlobalTensor<int64_t> gX_;
    GlobalTensor<int64_t> gY_;
    GlobalTensor<int64_t> gZ_;

    // Ping-pong buffers of the manual pipeline (GM <-> UB); their event ids are EVT_ID[0] / EVT_ID[1].
    TBuf<TPosition::VECIN> inBuf_[BUF_DEPTH];
    TBuf<TPosition::VECOUT> outBuf_[BUF_DEPTH];

    TBuf<> tLoIdx_;
    TBuf<> tInterIdx_;
    TBuf<> tMask_;

    TBuf<> tLoBuf_;
    TBuf<> tHiBuf_;
    TBuf<> tPack_;
    TBuf<> tP1Lo_;

    TBuf<> tMx0_;
    TBuf<> tMy0_;
    TBuf<> tMx1_;
    TBuf<> tMy1_;
    TBuf<> tMu_;
    TBuf<> tMv_;
    TBuf<> tMw_;
    TBuf<> tMa_;

    int64_t start_ = 0;
    int64_t end_ = 0;
    bool valid_ = false;
};
} // namespace

extern "C" __global__ __aicore__ void mul_ascend(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(MulAscendTilingData);
    GET_TILING_DATA(tilingData, tiling);

    MulAscendKernelPass op;
    op.Init(reinterpret_cast<__gm__ const int64_t*>(x),
            reinterpret_cast<__gm__ const int64_t*>(y),
            reinterpret_cast<__gm__ int64_t*>(z),
            static_cast<int64_t>(tilingData.size));
    op.Process();
}

