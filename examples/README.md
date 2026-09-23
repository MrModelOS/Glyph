# Примеры Glyph

Все примеры компилируются и запускаются последней версией компилятора.
Проверка сразу всех:

```bash
examples/run_all.sh          # OK/FAIL по каждому примеру, exit 1 при падении
```

Одиночный пример:

```bash
glyphc run --input examples/hello.glyph
```

| Файл | Тема |
|---|---|
| `hello.glyph` | вывод, строки, конкатенация `++` |
| `simple.glyph` | минимальная программа |
| `structs.glyph` | структуры, свободные функции |
| `impl_const.glyph` | `@const`, структуры |
| `methods.glyph` | методы через `@impl`: `p.norm()`, `p.scaled(2.0)` |
| `lists.glyph` | `List<T>`: литералы, индексация, `len`, `append`, итерация, `++`, диапазоны |
| `result_option.glyph` | `Result`/`Option` с данными: `Ok(x)`/`Some(x)`, `match` с payload |
| `enums.glyph` | перечисления с дискриминантами, `match` |
| `enum_test.glyph` | перечисления с данными, `match` с привязками |
| `enum_data.glyph` | варианты с данными, поля |
| `enum_payload.glyph` | вариант как контейнер разных типов |
| `match.glyph` | `match` с литералами и wildcard |
| `while.glyph` | цикл `while` |
| `v1_features.glyph` | `while`, `as`-касты, диапазоны `0..10`, списки, `match` |
| `math_test.glyph` | `std.math` без `@use` |
| `stdlib_test.glyph` | `std.io`, `std.string` |
| `error_handling.glyph` | `#guard` + перечисление-ошибка (`DivisionResult`) |
| `generics.glyph` | обобщённые функции: `<T>`, `<T, K>`, `Option<T>`, `Result<T, E>`, цепочки |
| `concurrency.glyph` | `@fn async`, `spawn`/`await`, `Channel<T>`, `send`/`recv`/`close` |
| `tests/` | тестовый проект: `@test`-модуль `calc.glyph` + `main.glyph`, `glyphc test` |
| `nns/mlp.ns` | NeuralScript: MLP classifier `Tensor[Batch, Features]` → `Dense`+`Dropout`, `train{grad{}}` with `cross_entropy` |
| `nns/transformer.ns` | NeuralScript: minimal transformer `Embedding→Attention→LayerNorm→MLP`, AOT autodiff demo |
| `nns/dense.ns` | NeuralScript: 7-block transformer with wide final FFN (`WIDE=20484`), same params as MoE K=9 |
| `nns/static.ns` / `nns/neumoe/static.ns` | NeuralScript: static MoE (K=9 from step 0), frozen routing |
| `nns/neumoe.ns` / `nns/neumoe/neumoe.ns` | NeuralScript: NeuMoE growing MoE (`ns_expert_birth`), lifecycle + `ns_expert_*` C-ABI |
| `nns/moe_growth.ns` | NeuralScript: MoE growth study variant |
| `nns/host.cpp` / `nns/neumoe/host.cpp` | C++ host drivers for `--runtime` — generic over `ns_runtime.h` C-ABI (see docs/nns.md) |

## Проекты (папки)

### tests/

```bash
cd examples/tests
glyphc test  # 9 passing checks + 1 intentional negative test
```

См. [docs/testing.md](../docs/testing.md).