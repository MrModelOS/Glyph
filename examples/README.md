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
| `lists.glyph` | типизированные `List<T>`: литералы, индексация примитивов и структур |
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
| `project/` | многофайловый проект: 6 модулей, импорт `@use`, вызовы `math::…` |
| `tests/` | тестовый проект: `@test`-модуль `calc.glyph` + `main.glyph`, `glyphc test` |

## Проекты (папки)

### project/

```bash
glyphc run --input examples/project/src/main.glyph
# 3 + 4 = 7.000000
# 3 * 4 = 12.000000
# 5^2 = 25.000000
# distance(0,0,3,4) = 5.000000
# Module system test passed!
```

Источники: `main.glyph` — entry, импортирует `math` и `geometry` через `@use`
и вызывает функции квалифицированно (`math::add`, `geometry::distance`).
Рядом лежат `a.glyph`/`b.glyph`/`cyclic.glyph` для проверки разрешения
циклических `@use`. Каждый модуль также можно запустить как entry.

### tests/

```bash
cd examples/tests
glyphc test
```

См. [docs/testing.md](../docs/testing.md).