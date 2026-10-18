# mul_ascend

## Примитивы, которые реально есть на `dav_c220` (CANN 9.1)

| Операция | Поддерживаемые типы | Что используется в ядре |
| --- | --- | --- |
| `Add`/`Sub`/`Mul` | `half`, `float`, `int16_t`, `int32_t` | `int32_t` (модульная арифметика 2³² эквивалентна `uint32_t`) |
| `And`/`Or`/`Xor`/`Not` | только `int16_t`, `uint16_t` | `uint16_t`-представление буферов, `count = 2 * elements` |
| `ShiftLeft`/`ShiftRight` | `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, **скалярная** величина сдвига | `uint32_t` + скалярные 16/31 (логический сдвиг) |
| `CreateVecIndex`/`Muls`/`Adds` | `int32_t` (и др.) | построение индексных векторов |
| `Gather` | `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `half`, `float`, … ; смещения — **байты** | разделение/сборка 32-битных половин |
| `DataCopyPad` | любой тип, длина в байтах | загрузка/выгрузка тайла произвольной длины |

## Алгоритм

Для каждого элемента, где `a = aLo + (aHi << 32)`, `b = bLo + (bHi << 32)`:

```text
a * b = (aLo * bLo) + ((aLo * bHi + aHi * bLo) << 32)      (mod 2^64)
```

1. **Split (de-interleave).** `x` и `y` грузятся в один буфер подряд, после чего **один** `Gather`
   с байтовыми смещениями `8*j` даёт `[aLo | bLo]`, а второй (`8*j + 4`) — `[aHi | bHi]`.
2. **Три произведения.** `rLo = Mul(aLo, bLo)` — это младшие 32 бита результата. Перекрёстные
   `p1 = Mul(aLo, bHi)` и `p2 = Mul(aHi, bLo)` нужны только младшими 32 битами, поэтому на каждое
   приходится ровно **одна** векторная инструкция.
3. **`MulHi(aLo, bLo)`** — старшие 32 бита `aLo*bLo` через 16-битное разложение, без битовых
   «переносных» формул:

   ```text
   x*y = A + T*2^16 + W*2^32,   A = x0*y0, T = x0*y1 + x1*y0, W = x1*y1
   U = x0*y1 = U1*2^16 + U0,  V = x1*y0 = V1*2^16 + V0
   s0 = U0 + V0                -> T_lo = s0 & 0xFFFF, c0 = s0 >> 16   (s0 < 2^17, точно)
   T_hi = U1 + V1 + c0          (точно равно T >> 16)
   high32 = W + T_hi + (((A >> 16) + T_lo) >> 16)                       (22 инструкции)
   ```

4. **Сборка.** `rHi = MulHi(aLo, bLo) + p1 + p2` (переполнение отбрасывается по модулю 2³² — это и
   есть семантика `int64` wrap-around).
5. **Merge (interleave).** `rLo` и `rHi` лежат в одном буфере (`tPack_`), и один `Gather` с
   индексом `4*(j/2) + (j%2)*4*TILE_ELEMS` собирает их обратно в layout `int64`.

Все индексы и маска строятся один раз в `Init` (`CreateVecIndex` + `Muls`/`Adds`/`Shift`/`Sub` +
`Duplicate`), тайл обрабатывается циклом `TILE_ELEMS = 512`, хвост идёт тем же путём через
`DataCopyPad`.

## Оптимизации (относительно первой рабочей версии)

| Что | Как | Эффект |
| --- | --- | --- |
| Перекрёстные произведения | `p1`/`p2` считаются одним `Mul` (младшие 32 бита), а не полным `32x32 -> 64` | 60 -> 2 инструкции |
| `high32(aLo*bLo)` | 16-битные полусуммы вместо `carry(a+b) = ((a & b) \| ((a \| b) & ~sum)) >> 31`, применённой дважды | 30 -> 22 инструкции |
| Split | объединённый входной буфер `[x | y]`: один `Gather` на младшие половины, один на старшие | 4 `Gather` -> 2 |
| Выделение 16-битных половин | маска `0x0000FFFF` (одна `Duplicate` в `Init`), `And` вместо пары сдвигов | -5 инструкций на тайл |
| Пайплайн | ручной double buffering (`TBuf` ping-pong + явные `SetFlag`/`WaitFlag<HardEvent>` на `PIPE_MTE2`/`PIPE_V`/`PIPE_MTE3`) | загрузка тайла `i+1`, вычисления тайла `i` и выгрузка тайла `i-1` перекрываются |
| Внутрипайповые барьеры | не нужны: `Gather`/`Mul`/`Add`/`And`/`Shift` — обычные in-order инструкции внутри `PIPE_V` (барьеры CANN ставит только вокруг reduce/transpose/select/bilinear). Опционально включаются `-DMUL_ASCEND_V_BARRIERS=1` (диагностика / A-B замер) | 0 барьеров в горячем пути |
| Буферы UB | переиспользование мёртвых буферов: один индексный вектор на оба `Gather` (`srcBaseOffset = 4`), скретч `Init` в `tPack_`/`tLoBuf_`, один скретч на два кросс-члена, скретч `MulHi` 13 -> 8 | 26 -> **17 буферов**, 88 -> **64 KB** UB |
| Размер тайла | `TILE_ELEMS` 256 -> 512 | меньше фиксированных издержек (синхронизации, индексы) на элемент |

Итого арифметика на тайл: **92 -> 27** инструкций, а вместе с `Gather` — **97 на 256 элементов
(0.38 на элемент) -> 30 на 512 элементов (0.06 на элемент)**, то есть ~6x меньше инструкций на
элемент. Из оставшихся 30 три — это `Gather` (самая «дорогая» инструкция), остальные 27 —
арифметика/сдвиги.

## Структура

* `op_kernel/mul_ascend.cpp` — ядро (`MulAscendKernelPass`) + точка входа `mul_ascend`.
* `op_kernel/mul_ascend_tiling.h` — tiling-структура.
* `op_host/mul_ascend.cpp` — tiling/infer-shape/infer-dtype + `OpDef`.
* `test/cpu_debug_check.cpp`, `test/build_cpu_debug.sh` — проверка ядра на CPU-модели 910B.
* `test/hw_check.cpp`, `test/build_hw_check.sh` — проверка опа на **реальном** 910B через aclnn
  (сверка всех элементов с эталоном + опциональный замер времени).
* `test/direct_launch_check.cpp`, `test/build_direct_launch.sh` — запуск того же ядра **без регистрации
  опа**: готовый `.o` грузится через `aclrtBinaryLoadFromFile` и стартует `aclrtLaunchKernel`
  (без aclnn, без `TilingFunc`, без установки пакета).
* `test/build_reference_check.sh`, `src/reference_check.cpp` — проверка математики на CPU.
* `docs/mul_ascend_algorithm_flow.svg` — схема алгоритма.
* `docs/mul_ascend_example_trace.svg` — пример преобразования данных с реальными значениями
  (дамп промежуточных буферов настоящего ядра на CPU-модели).
* `test/example_trace/` — инструментарий для этого примера: `patch.py` (инструментированная копия ядра),
  `main.cpp` (вход из 3 элементов и вывод дампа), `mkpic.py` (проверка трассы + отрисовка SVG), `run.sh`.
* `src/mul_int64_vec_kernel.cpp` — ранний набросок ядра; использует `DeInterleave`/`Interleave`,
  которых на `dav_c220` нет, и в сборку не входит (оставлен как история).

## Сборка CANN-пакета

```bash
cd /home/user/mul_ascend
bash build.sh          # -> build_out/custom_opp_ubuntu_x86_64.run
```

## Запуск на реальном железе (Ascend 910B)

```bash
# 1) собрать пакет кастомного опа
bash build.sh                                  # -> build_out/custom_opp_*.run

# 2) установить и выставить окружение
./build_out/custom_opp_*.run --install --install-path=$HOME/Ascend/custom_opp
export ASCEND_CUSTOM_OPP_PATH=$HOME/Ascend/custom_opp/vendors/customize

# 3) функциональная проверка на девайсе (aclnn-интерфейс опа)
bash test/build_hw_check.sh                    # 21 размер, случайные + экстремальные значения
bash test/build_hw_check.sh --bench            # + время ядра (us/op, ns/элемент)
bash test/build_hw_check.sh --size 1048576     # один размер
```

`test/build_hw_check.sh` сам находит установленный vendor-каталог (`$ASCEND_CUSTOM_OPP_PATH`,
`<CANN>/opp/vendors/customize`, `$HOME/Ascend/custom_opp/vendors/customize`, `/usr/local/Ascend/opp/...`,
либо распакованный пакет в `build_out/`), собирает `test/hw_check.cpp` против `libcust_opapi`
и запускает его. Коды возврата: `0` — всё сошлось, `1` — расхождения, `2` — ошибка ACL/опа,
`3` — на хосте нет NPU или драйверных библиотек (тогда скрипт прямо предлагает CPU-проверки).

Что проверяет `hw_check`: создаёт на девайсе тензоры `x`, `y`, `z` (int64, ND), вызывает
`aclnnMulAscendGetWorkspaceSize` / `aclnnMulAscend` и сверяет **все** элементы с эталоном `(uint64)x * (uint64)y`,
посчитанным на хосте. Размеры: 1, 2, 3, 5, 7, 15…17, 255…257, 511…513, 1000, 4095…4097, 65536, 100000, 2^20 —
так в проверку попадают хвосты и некратные размеры; значения — случайные плюс
`INT64_MIN/MAX`, ±1, 0 и границы 32-битных половин (`0x00000000FFFFFFFF`, `0xFFFFFFFF00000000`).

Для профилирования на железе (`msprof` есть в тулките):

```bash
msprof --application="test/hw_check --bench" --output=./prof_out
```

## Запуск без регистрации опа (kernel launch)

```bash
bash build.sh                                     # нужен только сам бинарь ядра
bash test/build_direct_launch.sh                  # 21 размер
bash test/build_direct_launch.sh --bench          # + время ядра (us/op, ns/элемент)
bash test/build_direct_launch.sh --size 100000    # один размер
bash test/build_direct_launch.sh --kernel <path.o> --name <kernel>
```

Оператор при этом **не регистрируется** в CANN: пакет `custom_opp_*.run` не устанавливается,
`ASCEND_CUSTOM_OPP_PATH` не нужен, `aclnn`-обёртка и `TilingFunc` не вызываются — используется только
device-бинарь ядра из `build_out/op_kernel/ascendc_kernels/binary/ascend910b/mul_ascend/`.

`test/build_direct_launch.sh` сам находит этот `.o`, вытаскивает имя ядра из ELF-символа
(`MulAscend_<hash>_0`), собирает `test/direct_launch_check.cpp` против `libascendcl` и запускает его.
Программа делает ровно то, что делает официальный `mskl` (MindStudio kernel launch), только на C++:

1. `aclrtBinaryLoadFromFile` — грузит device-ELF как есть; секция `.ascend.meta.<kernel>` (тип ядра
   `K_TYPE_AIV` и описание аргументов) в нём уже есть, драйвер читает её сам;
2. `aclrtBinaryGetFunction(bin, "<kernel>")`, при неудаче — `aclrtBinaryGetFunctionByEntry(bin, 0)`;
3. запуск через `aclrtLaunchKernel(func, blockDim, args, 40, stream)`: аргументы — это ровно пять
   `GM_ADDR` точки входа `mul_ascend(x, y, z, workspace, tiling)`, то есть 40 байт (столько же
   заявляет секция `__CCE_KernelArgSize` самого бинаря);
4. `tiling` указывает на device-буфер, в котором лежит то, что записал бы `TilingFunc` —
   `uint32` число элементов по смещению 0 (`op_kernel/mul_ascend_tiling.h`); ядро читает tiling из GM
   (у опа динамическая форма), поэтому размер задаётся хостом на каждый запуск, а `block-dim` = 16
   берётся из `context->SetBlockDim(16)`;
5. результат сверяется поэлементно с эталоном `(uint64)x * (uint64)y`.

Коды возврата те же, что у остальных проверок: `0` — всё сошлось, `1` — расхождения, `2` — ошибка ACL,
`3` — на хосте нет NPU или драйверных библиотек (скрипт сразу предлагает CPU-проверки).

## Локальная проверка (без NPU)

```bash
# 1) математика 32x32 -> 64 (200000 случайных пар)
bash test/build_reference_check.sh

# 2) ядро целиком на CPU-модели Ascend910B (tools/cpudebug)
bash test/build_cpu_debug.sh

# 3) трассировка примера и перерисовка docs/mul_ascend_example_trace.svg
bash test/example_trace/run.sh
```

`test/build_cpu_debug.sh` компилирует `op_kernel/mul_ascend.cpp` с `ASCENDC_CPU_DEBUG=1`
(хедеры `compiler/tikcpp/tikcfw`, `__NPU_ARCH__=2201`) и запускает реальную точку входа
`mul_ascend` на `libcpudebug.so` для `Ascend910B1`, сверяя результат с `uint64`-произведением.

