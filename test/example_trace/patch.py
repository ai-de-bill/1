import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'op_kernel', 'mul_ascend.cpp')
DST = os.path.join(HERE, 'mul_ascend_dbg.cpp')

src = io.open(SRC, encoding='utf-8').read()

def rep(old, new, cnt=1):
    global src
    assert src.count(old) == cnt, (src.count(old), old[:60])
    src = src.replace(old, new, cnt)

# globals for the dump pointer (test only)
rep('using namespace AscendC;\n',
    'using namespace AscendC;\n\n// ==== DBG instrumentation (test-only copy) ====\n'
    'static uint32_t* g_dbgDump = nullptr;\nstatic constexpr int32_t DBG_LANES = 8;\n')

# dump helper inside the class
rep('    __aicore__ inline void BuildTables()',
    '    __aicore__ inline void DbgSlot(int32_t slot, const LocalTensor<uint32_t>& t, int32_t off)\n'
    '    {\n'
    '        if (g_dbgDump == nullptr) { return; }\n'
    '        PipeBarrier<PIPE_ALL>();\n'
    '        GlobalTensor<uint32_t> gd;\n'
    '        gd.SetGlobalBuffer((__gm__ uint32_t*)g_dbgDump + slot * DBG_LANES, DBG_LANES);\n'
    '        DataCopyPad(gd, t[off], DataCopyExtParams{1, DBG_LANES * WORD_BYTES, 0, 0, 0});\n'
    '    }\n\n'
    '    __aicore__ inline void BuildTables()')

# raw tile + the two halves of both operands
rep('        auto aLo = loBuf;\n',
    '        DbgSlot(4, in32, 0);\n'
    '        DbgSlot(0, loBuf, 0);\n'
    '        DbgSlot(1, loBuf, TILE_ELEMS);\n'
    '        DbgSlot(2, hiBuf, 0);\n'
    '        DbgSlot(3, hiBuf, TILE_ELEMS);\n'
    '        auto aLo = loBuf;\n')

# rLo / rHi are final right before the merge gather
rep('        Gather(out32, pack, tInterIdx_.Get<uint32_t>(), 0, static_cast<uint32_t>(2 * cnt));\n',
    '        DbgSlot(5, pack, 0);\n'
    '        DbgSlot(6, pack, TILE_ELEMS);\n'
    '        Gather(out32, pack, tInterIdx_.Get<uint32_t>(), 0, static_cast<uint32_t>(2 * cnt));\n')

# MulHi internals: all four products are ready after the W multiply; afterwards the operand buffers
# are reused (x0 -> U0, y0 -> V0, x1 -> s0 -> T_lo, y1 -> c0, u -> U1 -> T_hi).
rep('// W = x1 * y1  (x1/y1 are dead after this)\n',
    '// W = x1 * y1  (x1/y1 are dead after this)\n'
    '        DbgSlot(8, x0, 0); DbgSlot(9, y0, 0); DbgSlot(10, x1, 0); DbgSlot(11, y1, 0);\n'
    '        DbgSlot(12, u, 0); DbgSlot(13, v, 0); DbgSlot(14, w, 0); DbgSlot(15, a, 0);\n')

rep('// u now holds T_hi\n',
    '// u now holds T_hi\n'
    '        DbgSlot(16, x0, 0); DbgSlot(17, y0, 0); DbgSlot(18, x1, 0);\n'
    '        DbgSlot(19, y1, 0); DbgSlot(20, u, 0);\n')

rep('// + S_hi\n    }\n',
    '// + S_hi\n'
    '        DbgSlot(21, a, 0);   // a currently holds S_hi\n    }\n')

io.open(DST, 'w', encoding='utf-8').write(src)
print('patched kernel written, DbgSlot calls =', src.count('DbgSlot('))
