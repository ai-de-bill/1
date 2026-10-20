# hw_test — MulAscend на реальном 910B без регистрации опа

Самодостаточный каталог: нужен только установленный **CANN-тулкит** (и драйвер NPU на целевой машине).
Оператор в CANN не регистрируется: device-бинарь ядра загружается и запускается напрямую.

## Что нужно на сервере с девайсом (чек-лист)

| Что | Зачем / как проверить |
| --- | --- |
| NPU Ascend 910B и **драйвер** | `npu-smi info` должен показать устройство; библиотеки драйвера лежат в `/usr/local/Ascend/driver/lib64` (`libascend_hal.so`) — без них `run.sh` выйдет с кодом 3 |
| Доступ к устройству | у пользователя должны быть права на `/dev/davinci*` (обычно группа `HwHiAiUser`: `sudo usermod -aG HwHiAiUser $USER`, затем перелогиниться) |
| **CANN-тулкит** | `export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest` (скрипт умеет искать сам). Наша сборка сделана на CANN 9.1.0: если на сервере другая версия — надёжнее пересобрать ядро на месте: `bash run.sh --build-kernel` |
| `g++` и `readelf` | хост-программа собирается на месте; `readelf` (binutils) нужен скрипту, чтобы прочитать имя ядра и размер аргументов из `.o` |
| сам каталог `hw_test/` | `scp -r hw_test user@server:~/` — для запуска достаточно `run.sh` + `direct_launch_check.cpp` + `prebuilt/`; `kernel/` нужен только для пересборки ядра |

Регистрировать что-либо в CANN не нужно: ни `custom_opp_*.run`, ни `ASCEND_CUSTOM_OPP_PATH` — рантайм
загружает `.o` напрямую (`rtRegisterAllKernel` + `rtKernelLaunchWithHandleV2`).

**Проверка по шагам:**

```bash
ssh user@server
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
cd ~/hw_test
bash run.sh --size 512      # быстрый smoke-тест: ожидаем "n=512 OK"
bash run.sh                 # все 21 размер, сверка с эталоном
bash run.sh --bench 50      # + время ядра (us/op, ns/элемент)
```

Если сервер на другом чипе (не `ascend910b`) — пересоберите ядро под него: `bash run.sh --build-kernel`
(при необходимости поправив `AddConfig("ascend910b")` в `kernel/op_host/mul_ascend.cpp`), либо подайте
свой `.o` через `--kernel`.

> `prebuilt/` содержит бинарь текущей ревизии ядра. Tiling читается ядром из GM-указателя так же, как
> `x/y/z` (`tiling[0]` — число элементов, `uint32`), поэтому размер задаётся хостом на каждый запуск.
> Если правили `kernel/op_kernel/*.cpp` — соберите `--build-kernel` (иначе `run.sh` предупредит, что
> бинарь старше исходников).

## Состав

| Путь | Что это |
| --- | --- |
| `direct_launch_check.cpp` | хост: грузит `.o` ядра, запускает его и сверяет **все** элементы с эталоном |
| `run.sh` | сборка хоста (и опционально ядра) + запуск |
| `kernel/` | минимальный проект опа (`op_host` + `op_kernel`) — нужен, только чтобы собрать/поправить ядро |
| `prebuilt/` | уже собранный `MulAscend_<hash>.o` (+ `.json`) — его достаточно для запуска |

## Быстрый старт

```bash
# 1) скопировать каталог на сервер с 910B
scp -r hw_test user@server:~/ && ssh user@server

# 2) выставить тулкит, если ещё не выставлен
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

# 3) запустить
cd ~/hw_test
bash run.sh                     # все 21 размер (сверка с эталоном)
bash run.sh --size 100000       # один размер
bash run.sh --bench 20          # + время ядра (us/op, ns/элемент)
```

Ожидаемый вывод:

```
toolkit    : /usr/local/Ascend/ascend-toolkit/latest
kernel .o  : .../hw_test/prebuilt/MulAscend_96d2ee1256361ca6f1fd7aed5dc397bc.o
kernel name: MulAscend_96d2ee1256361ca6f1fd7aed5dc397bc_0
args size  : 40 bytes (x, y, z, workspace, tiling)
building   : .../hw_test/direct_launch_check
launch     : rtRegisterAllKernel (magic=0x41415246) + rtKernelLaunchWithHandleV2(tilingKey=0)
  n=1        OK
  ...
ALL PASSED (21 sizes, direct kernel launch)
exit code  : 0   (0 = ok, 1 = wrong results, 2 = acl error, 3 = no device)
```

## Что именно запускается

1. `rtRegisterAllKernel` — регистрирует device-ELF в рантайме **как есть** (`magic = 0x41415246` =
   `RT_DEV_BINARY_MAGIC_ELF_AIVEC` для векторного ядра dav-c220). Секция `.ascend.meta.<kernel>` (тип ядра
   и описание аргументов) в бинаре уже есть — драйвер читает её сам.
2. `rtKernelLaunchWithHandleV2(handle, tilingKey=0, blockDim=16, &argsEx, …, stream, …)` с `hasTiling = 0`.
   Аргументы — пять `GM_ADDR` точки входа `mul_ascend(x, y, z, workspace, tiling)` = 40 байт (столько же
   заявляет секция `__CCE_KernelArgSize` бинаря). `tiling` — обычный указатель на device-буфер, в котором
   лежит `uint32` число элементов по смещению 0 (то, что записал бы `TilingFunc`), поэтому размер задаётся
   хостом на каждый запуск; `block-dim` = 16 (`context->SetBlockDim(16)` в `op_host`).
3. Ничего из фреймворка опа в рантайме не участвует: нет установки `custom_opp_*.run`, нет
   `ASCEND_CUSTOM_OPP_PATH`, нет `aclnn`-обёртки, нет вызова `TilingFunc`.
4. Проверка: **все** элементы результата сверяются с эталоном `(uint64)x * (uint64)y` (умножение с
   естественным переполнением, mod 2^64). Данные — случайные плюс `INT64_MIN`/`INT64_MAX`; размеры:
   1, 2, 3, 5, 7, 15…17, 255…257, 511…513, 1000, 4095…4097, 65536, 100000, 2^20.

## Пересборка ядра из исходников

```bash
bash run.sh --build-kernel --size 1000     # сначала bash kernel/build.sh, потом запуск
```

`kernel/` — обычный проект кастомного опа (как в репозитории): `bash kernel/build.sh` собирает `.o` в
`kernel/build_out/op_kernel/ascendc_kernels/binary/ascend910b/mul_ascend/`. Свой бинарь можно подать
напрямую: `bash run.sh --kernel /path/to/MulAscend_*.o`.

> `run.sh` ядро **не пересобирает**: он берёт готовый `.o` (из `kernel/build_out/`, иначе из `prebuilt/`).
> Если правили `kernel/op_kernel/*.cpp` — используйте `--build-kernel`; скрипт предупредит, если исходник
> новее бинаря (`WARNING: ... is newer than ...`).

## Опции `run.sh`

| Опция | Смысл |
| --- | --- |
| `--size N` | прогнать один размер вместо списка из 21 |
| `--bench [iters]` | замер времени ядра (по умолчанию 50 итераций) |
| `--kernel PATH` | использовать указанный `.o` вместо `kernel/build_out` / `prebuilt` |
| `--build-kernel` | сначала собрать ядро: `bash kernel/build.sh` (лог: `kernel_build.log`) |
| `--check-inputs` | диагностика: device-адреса буферов + обратное чтение `x` с девайса |
| `--block-dim N` | число блоков (по умолчанию 16, как в `SetBlockDim(16)`) |
| `--device N` | номер устройства (по умолчанию 0) |
| `--entry N` | tiling key для `rtKernelLaunchWithHandleV2` (по умолчанию 0) |
| `--sim` / `--sim-soc NAME` | запуск на **camodel-симуляторе** тулкита вместо железа (по умолчанию `Ascend910B1`) |
| `--help` | краткая справка |

## Коды возврата

`0` — всё сошлось · `1` — расхождения в результатах · `2` — ошибка ACL/сборки · `3` — на машине нет NPU
или драйверных библиотек.

## Если что-то не так

| Симптом | Что делать |
| --- | --- |
| `ERROR: the ACL runtime cannot be loaded (missing driver libraries)` + `exit 3` | на машине нет драйвера NPU: проверьте `npu-smi info` и `/usr/local/Ascend/driver/lib64`; без железа — `--sim` |
| `[FAIL] rtRegisterAllKernel: ret=…` | бинарь не подходит этой версии CANN/чипу — пересоберите: `--build-kernel`; `magic` для AIV-ядра `0x41415246` (по умолчанию) |
| `WARNING: ... is newer than ...` | бинарь устарел → `--build-kernel` |
| `FAILED`, в выводе нули и `the kernel never wrote the output` | ядро не записало выход — проверьте, что не правили ядро без пересборки, и что используется rt-путь (по умолчанию) |
| `FAILED` на конкретных размерах | сравните `got`/`expected` из вывода (печатаются первые 8 элементов) — это уже вопрос к математике ядра, а не к запуску |
| `--sim` идёт долго | camodel циклоточный: берите `--size 512` / `--size 4097`, полный список идёт десятки минут |

## Где это уже проверено

* **camodel-симулятор NPU** (CANN 9.1.0, Ascend910B1): `ALL PASSED (21 sizes)`, включая `65536`,
  `100000`, `2^20`; лог модели подтверждает исполнение 16 векторных ядер.
* **CPU-модель 910B** (`tools/cpudebug`): `ALL PASSED (22 sizes)`.
* На реальном железе запуск не проверялся (в среде разработки нет NPU/драйвера) — это и есть цель этого
  каталога. Если что-то пойдёт не так, вывод `run.sh` (строки `launch:`, `[FAIL] …`, `got/expected`)
  содержит всё нужное для диагностики.
