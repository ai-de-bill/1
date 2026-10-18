#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Builds docs/mul_ascend_example_trace.svg from the real CPU-model dump of the kernel.

trace.txt comes from running an instrumented copy of op_kernel/mul_ascend.cpp on libcpudebug
(Ascend910B1) with a 3-element input; every intermediate buffer is dumped by the device-side code.
This script re-derives all values independently (as a check) and then draws the picture.
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'trace.txt')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, '..', '..', 'docs', 'mul_ascend_example_trace.svg')
M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF


def parse(path):
    txt = io.open(path, encoding='utf-8').read()
    i = re.search(r'INPUT x = (\S+) (\S+) (\S+)\nINPUT y = (\S+) (\S+) (\S+)', txt)
    zo = re.search(r'OUT z = (\S+) (\S+) (\S+)', txt)
    refs = re.findall(r'REF z\[\d\] = (\S+)', txt)
    slots = {}
    for m in re.finditer(r'SLOT (\d+) \|(.*)\|\s*((?:[0-9a-f]{8} ?)+)', txt):
        slots[int(m.group(1))] = [int(w, 16) for w in m.group(3).split()]
    return ([int(v, 16) for v in i.group(1, 2, 3)], [int(v, 16) for v in i.group(4, 5, 6)],
            [int(v, 16) for v in zo.group(1, 2, 3)], [int(v, 16) for v in refs], slots)


def mulhi(aLo, bLo):
    """Same op sequence as MulHi() in the kernel."""
    x0, x1 = aLo & 0xFFFF, aLo >> 16
    y0, y1 = bLo & 0xFFFF, bLo >> 16
    U, V, W, A = x0 * y1, x1 * y0, x1 * y1, x0 * y0
    u0, v0 = U & 0xFFFF, V & 0xFFFF
    s0 = u0 + v0
    c0, Tlo = s0 >> 16, s0 & 0xFFFF
    Thi = (U >> 16) + (V >> 16) + c0
    S = (A >> 16) + Tlo
    Shi = S >> 16
    return dict(x0=x0, y0=y0, x1=x1, y1=y1, U=U & M32, V=V & M32, W=W & M32, A=A & M32, U0=u0, V0=v0,
                Tlo=Tlo, c0=c0, Thi=Thi, Shi=Shi, hi=(W + Thi + Shi) & M32)


def verify(x, y, z, ref, s, n=3):
    bad = []

    def chk(name, got, want):
        if got != want:
            bad.append('%s dump=%08x expect=%08x' % (name, got, want))

    for i in range(n):
        a, b = x[i], y[i]
        aLo, aHi, bLo, bHi = a & M32, a >> 32, b & M32, b >> 32
        m = mulhi(aLo, bLo)
        chk('aLo[%d]' % i, s[0][i], aLo)
        chk('bLo[%d]' % i, s[1][i], bLo)
        chk('aHi[%d]' % i, s[2][i], aHi)
        chk('bHi[%d]' % i, s[3][i], bHi)
        for k, slot in (('x0', 8), ('y0', 9), ('x1', 10), ('y1', 11), ('U', 12), ('V', 13), ('W', 14),
                        ('A', 15), ('U0', 16), ('V0', 17), ('Tlo', 18), ('c0', 19), ('Thi', 20), ('Shi', 21)):
            chk('%s[%d]' % (k, i), s[slot][i], m[k])
        p1, p2 = (aLo * bHi) & M32, (aHi * bLo) & M32
        rLo, rHi = (aLo * bLo) & M32, (m['hi'] + p1 + p2) & M32
        chk('rLo[%d]' % i, s[5][i], rLo)
        chk('rHi[%d]' % i, s[6][i], rHi)
        chk('z[%d]' % i, z[i], (rHi << 32) | rLo)
        chk('ref[%d]' % i, z[i], (a * b) & M64)
    return bad


# ------------------------------------------------------------------ rendering
BLUE, GREEN, YELL, VIOL, CYAN, OKC = '#bfdbfe', '#bbf7d0', '#fde68a', '#ddd6fe', '#a5f3fc', '#86efac'


def esc(t):
    """XML-escape for text payloads (labels contain & and >>)."""
    return t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def cell(x, y, t, fill, w=110, h=30, fs=14):
    return ('<rect x="%d" y="%d" width="%d" height="%d" rx="6" fill="%s" stroke="#1f2937" '
            'stroke-width="1.4"/><text x="%d" y="%d" text-anchor="middle" class="mono" '
            'font-size="%dpx">%s</text>' % (x, y, w, h, fill, x + w / 2, y + h / 2 + 5, fs, esc(t)))


def txt(x, y, t, cls='text'):
    return '<text x="%d" y="%d" class="%s">%s</text>' % (x, y, cls, esc(t))


def row(y, lab, vals, fill, w=110, fmt='%08x', cx=380, fs=14, h=30):
    out = [txt(30, y + 20, lab)]
    for k, v in enumerate(vals):
        out.append(cell(cx + k * (w + 10), y, fmt % v, fill, w=w, h=h, fs=fs))
    return '\n  '.join(out)


def sec(y, title):
    rule = '<line x1="30" y1="%d" x2="1410" y2="%d" stroke="#94a3b8" stroke-width="1.5"/>' % (y, y)
    return rule + '\n  ' + txt(30, y + 26, title, 'stage')


def arr(y, t=''):
    s = '<line x1="700" y1="%d" x2="700" y2="%d" class="arrow"/>' % (y, y + 24)
    if t:
        s += '\n  ' + txt(716, y + 18, t, 'small')
    return s


def svg_head():
    return '''  <defs>
    <style>
      .title { font: 700 27px 'DejaVu Sans', Arial, sans-serif; fill: #0b1f3a; }
      .subtitle { font: 500 15px 'DejaVu Sans', Arial, sans-serif; fill: #2c3e50; }
      .stage { font: 700 18px 'DejaVu Sans', Arial, sans-serif; fill: #0b1f3a; }
      .text { font: 500 15px 'DejaVu Sans', Arial, sans-serif; fill: #1f2937; }
      .mono { font: 500 14px 'DejaVu Sans Mono', monospace; fill: #111827; }
      .small { font: 500 13px 'DejaVu Sans', Arial, sans-serif; fill: #374151; }
      .arrow { stroke: #334155; stroke-width: 2.6; fill: none; marker-end: url(#arrowhead); }
    </style>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#334155"/>
    </marker>
  </defs>'''


def main():
    x, y, z, ref, s = parse(TRACE)
    bad = verify(x, y, z, ref, s)
    if bad:
        print('TRACE MISMATCH (%d):' % len(bad))
        for b in bad:
            print('  ', b)
        return 1

    L = ['<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="1290" viewBox="0 0 1440 1290">',
         svg_head(),
         '  <rect x="0" y="0" width="1440" height="1290" fill="#ffffff"/>',
         '  ' + txt(24, 38, 'mul_ascend: что происходит с данными на реальном примере', 'title'),
         '  ' + txt(24, 62, 'дамп промежуточных буферов из настоящего ядра на CPU-модели Ascend910B; '
                            '3 элемента (хвост тайла), показаны только валидные лейны', 'subtitle')]

    yy = 78
    L.append('  ' + sec(yy, '1) GM: входные int64 (x и y)')); yy += 36
    L.append('  ' + row(yy, 'x[i]', x, BLUE, w=200, fmt='%016x'))
    L.append('  ' + row(yy + 38, 'y[i]', y, BLUE, w=200, fmt='%016x'))
    L.append('  ' + txt(380, yy + 90, 'старшие 32 бита каждого int64 — это aHi/bHi, младшие — aLo/bLo', 'small'))
    yy += 100
    L.append('  ' + arr(yy)); yy += 32

    L.append('  ' + sec(yy, '2) UB: сырой тайл как uint32-лейны (x и y лежат подряд в одном буфере)'))
    yy += 36
    raw = []
    for i in range(3):
        raw += [x[i] & M32, x[i] >> 32]
    L.append('  ' + row(yy, 'x: lo0 hi0 lo1 hi1 ...', raw, BLUE, w=110))
    L.append('  ' + txt(380, yy + 44, 'y начинается по смещению 8*TILE_ELEMS = 4096 байт', 'small'))
    yy += 54
    L.append('  ' + arr(yy, 'Gather со смещениями В БАЙТАХ: 8j (младшие половины) и 8j+4 (старшие)'))
    yy += 32

    L.append('  ' + sec(yy, '3) Split: 2 Gather вместо 4 — сразу [aLo | bLo] и [aHi | bHi]')); yy += 36
    L.append('  ' + row(yy, 'aLo', [s[0][i] for i in range(3)], GREEN, w=110))
    L.append('  ' + row(yy + 38, 'aHi', [s[2][i] for i in range(3)], GREEN, w=110))
    L.append('  ' + row(yy, 'bLo', [s[1][i] for i in range(3)], GREEN, w=110, cx=790))
    L.append('  ' + row(yy + 38, 'bHi', [s[3][i] for i in range(3)], GREEN, w=110, cx=790))
    yy += 76
    L.append('  ' + arr(yy)); yy += 32

    L.append('  ' + sec(yy, '4) Три произведения: младшие 32 бита — ровно одна инструкция Mul')); yy += 36
    L.append('  ' + row(yy, 'rLo = Mul(aLo,bLo)', [s[5][i] for i in range(3)], YELL, w=130))
    L.append('  ' + row(yy + 38, 'p1 = Mul(aLo,bHi)', [(s[0][i] * s[3][i]) & M32 for i in range(3)], YELL, w=130))
    L.append('  ' + row(yy + 76, 'p2 = Mul(aHi,bLo)', [(s[2][i] * s[1][i]) & M32 for i in range(3)], YELL, w=130))
    L.append('  ' + txt(790, yy + 20, 'rLo — это сразу младшие 32 бита результата', 'small'))
    L.append('  ' + txt(790, yy + 58, 'p1/p2 нужны только младшими 32 битами', 'small'))
    yy += 116
    L.append('  ' + arr(yy, 'high32(aLo*bLo) = MulHi (22 инструкции), затем rHi = MulHi + p1 + p2'))
    yy += 32

    i = 1
    m = mulhi(s[0][i], s[1][i])
    L.append('  ' + sec(yy, '5) MulHi внутри — элемент [1] (все 0xFFFFFFFF: максимум переносов)'))
    yy += 36
    L.append('  ' + row(yy, 'x0  x1  y0  y1  (16 бит)', [m['x0'], m['x1'], m['y0'], m['y1']], VIOL, w=130))
    L.append('  ' + txt(380 + 4 * 140, yy + 20, 'x0/y0 — младшие 16 бит, x1/y1 — старшие', 'small'))
    L.append('  ' + row(yy + 38, 'U=x0*y1  V=x1*y0  W=x1*y1  A=x0*y0',
                        [m['U'], m['V'], m['W'], m['A']], VIOL, w=130))
    L.append('  ' + row(yy + 76, 'U0  V0  T_lo=s0&0xffff  c0=s0>>16',
                        [m['U0'], m['V0'], m['Tlo'], m['c0']], VIOL, w=130))
    L.append('  ' + txt(380 + 4 * 140, yy + 96, 'c0=1: перенос из s0 = U0+V0 = 0x0001+0x0001 = 0x10000', 'small'))
    L.append('  ' + row(yy + 114, 'T_hi  S_hi  high32', [m['Thi'], m['Shi'], m['hi']], VIOL, w=130))
    L.append('  ' + txt(380 + 3 * 140, yy + 134,
                        'T_hi = U1+V1+c0 = fffe+fffe+1 = 0x1fffc;  high32 = W+T_hi+S_hi = fffffffe', 'small'))
    yy += 152
    L.append('  ' + arr(yy)); yy += 32

    L.append('  ' + sec(yy, '6) tPack = [rLo | rHi]  и  Merge (Gather по 4*(j/2) + (j%2)*4*TILE_ELEMS)'))
    yy += 36
    L.append('  ' + row(yy, 'rLo в tPack', [s[5][j] for j in range(3)], CYAN, w=110))
    L.append('  ' + row(yy + 38, 'rHi в tPack', [s[6][j] for j in range(3)], CYAN, w=110))
    L.append('  ' + txt(790, yy + 20, 'rLo по смещению 0', 'small'))
    L.append('  ' + txt(790, yy + 58, 'rHi по смещению 512 лейнов = 2048 байт', 'small'))
    yy += 76
    L.append('  ' + row(yy, 'индекс merge (байты)', [0, 2052, 4, 2056, 8, 2060], '#e2e8f0', w=110))
    L.append('  ' + row(yy + 38, 'out32 (лейны)',
                        [s[5][0], s[6][0], s[5][1], s[6][1], s[5][2], s[6][2]], CYAN, w=110))
    L.append('  ' + txt(790, yy + 58, 'rLo и rHi встают в один int64', 'small'))
    yy += 80
    L.append('  ' + arr(yy)); yy += 32

    L.append('  ' + sec(yy, '7) GM: выходные int64 — совпадает с x[i] * y[i] mod 2^64'))
    yy += 36
    L.append('  ' + row(yy, 'z[i]', z, OKC, w=200, fmt='%016x'))
    L.append('  ' + row(yy + 38, 'эталон', ref, '#dcfce7', w=200, fmt='%016x'))
    L.append('  ' + txt(380, yy + 92,
                        'проверено моделью и независимым пересчётом в этом скрипте (0 расхождений)', 'small'))
    L.append('</svg>')
    io.open(OUT, 'w', encoding='utf-8').write('\n  '.join(L))
    print('OK: trace verified, svg written to', OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())

