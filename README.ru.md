# Glyph Language Compiler (glyphc)

[English](README.md) | Русский

[![CI](https://github.com/MrModelOS/Glyph/actions/workflows/ci.yml/badge.svg)](https://github.com/MrModelOS/Glyph/actions/workflows/ci.yml)
![version](https://img.shields.io/badge/glyphc-v1.3.0-blue)

Компилятор языка программирования **Glyph** v1.3.0, написанный на Rust.

Glyph транслируется в C-код (GNU statement expressions) и собирается через GCC или clang.
Скомпилированные программы — обычные нативные бинарники.

```
┌──────────┐   glyphc    ┌────────────┐   gcc/clang   ┌─────────┐
│ .glyph   │ ──────────▶ │     .c     │ ─────────────▶ │ binary  │
└──────────┘   (Rust)    └────────────┘               └─────────┘
```

## Возможности

- Статическая типизация: `Int64`, `UInt64`, `Float64`, `Bool`, `String`, `Bytes`
- Пользовательские типы: структуры (`@struct`), перечисления (`@enum`) с данными в вариантах
- Методы через `@impl`: `obj.method(args)`, ресивер — первый параметр
- Контроль потока: `if/else`, `match` (в т.ч. с паттернами вариантов и данными), `while`, `loop`, `for .. in`
- Контракты-клозы: `#guard(cond) else { ... };`
- Модули: `@module`, `@use`, `@pub`, квалифицированные вызовы `math::sqrt`
- `@const` — именованные константы (компилируются в `static const`)
- Типизированные списки `List<T>`: литералы `[..]`, индексация `arr[i]`, срезы `arr[a..b]`/`arr[a..=b]`,
  диапазоны `0..10`, длина `.len()`, `append`, итерация `for x in xs`, конкатенация `++`,
  `==` для POD-списков, `xs.free()`
- `drop(box)` — явное освобождение `Option`/`Result`; временные боксы в `match` освобождаются сами
- `Result`/`Option` с данными: `Result::Ok(x)`/`Option::Some(x)` — построение и
  сопоставление (`Ok(v)`/`Some(v)` в `match`) с payload, в т.ч. из `parse_int`/`parse_float`
- Обобщённые функции: `@fn identity<T>(x: T) -> T` — мономорфизация, вывод типов
  по аргументам и из аннотации `let`, вложенные типы (`Option<T>`, `List<T>`)
- Конкурентность (pthreads): `@fn async`, ленивые хендлы `Async<T>`, `spawn`/`await`,
  типизированные каналы `Channel<T>(capacity)` (буферизованные + rendezvous), `send`/`recv`/`close`
- Встроенный тестовый фреймворк: `@test`, ассерты, `glyphc test`

## Установка

Нужен Rust (1.70+), а также `gcc` (или `clang`) линкер в `PATH`.

```bash
git clone https://github.com/MrModelOS/Glyph.git
cd Glyph/glyphc
cargo build --release
```

Бинарь появится в `target/release/glyphc`. Для быстрых итераций:

```bash
cargo build
```

## Быстрый старт

`hello.glyph`:

```glyph
@fn main() -> Void {
    let name: String = "World";
    let greeting: String = "Hello, " ++ name ++ "!";
    print(greeting);
}
```

```bash
glyphc run --input hello.glyph
# Hello, World!
```

## CLI

| Подкоманда | Описание |
|---|---|
| `compile --input f.glyph [-o out.c] [--emit-ir] [--no-typecheck]` | Транслировать в C (по умолчанию `output.c`) |
| `check --input f.glyph` | Проверить синтаксис и типы без генерации кода |
| `tokens --input f.glyph` | Показать поток токенов |
| `ast --input f.glyph` | Показать AST |
| `run --input f.glyph [--compiler gcc] [--opt -O2]` | Скомпилировать и запустить |
| `build [--profile dev\|release]` | Собрать проект по `glyph.toml` |
| `test [-i f.glyph] [--compiler gcc] [--opt -O2]` | Запустить `@test`-функции (без `-i` сканирует `./src`) |
| `glyphc --lsp` | Экспериментальный LSP-сервер |

Общие флаги: `-h/--help`, `-V/--version`.

## Язык

Подробный справочник — [docs/language.md](docs/language.md). Краткая выжимка:

### Переменные и константы

```glyph
let count: Int64 = 42;          // неизменяемая
let mut total: Float64 = 0.0;   // изменяемая
@const PI: Float64 = 3.141592653589793;
```

### Структуры

```glyph
@struct Point {
    x: Float64,
    y: Float64,
}

@fn distance(a: Point, b: Point) -> Float64 {
    let dx: Float64 = b.x - a.x;
    let dy: Float64 = b.y - a.y;
    return sqrt(dx * dx + dy * dy);
}
```

Методы через `@impl` — ресивер передаётся первым параметром:

```glyph
@impl Point {
    @fn norm(p: Point) -> Float64 {
        return sqrt(p.x * p.x + p.y * p.y);
    }
}

@fn main() -> Void {
    let p: Point = Point { x: 3.0, y: 4.0 };
    let n: Float64 = p.norm();   // → Point_norm(p)
    print_float(n);
}
```

### Перечисления

````glyph
@enum Shape {
    Circle(Float64),          // вариант с данными
    Rectangle(Float64, Float64),
    Point,                    // без данных
}

@fn area(s: Shape) -> Float64 {
    match s {
        | Shape::Circle(r) => 3.14159 * r * r
        | Shape::Rectangle(w, h) => w * h
        | Shape::Point => 0.0
    }
}

@fn main() -> Void {
    let circle: Shape = Shape::Circle(5.0);
    print_float(area(circle));
}
````

### Обработка ошибок через `#guard`

```glyph
@enum DivisionResult {
    Success(Float64),
    DivisionByZero,
}

@fn divide(a: Float64, b: Float64) -> DivisionResult {
    #guard(b != 0.0) else {
        return DivisionResult::DivisionByZero;
    };
    return DivisionResult::Success(a / b);
}
```

### Модули

```glyph
// math.glyph
@module my.math
@pub @fn add(a: Int64, b: Int64) -> Int64 { return a + b; }

// main.glyph
@use my.math;
@fn main() -> Void {
    let sum: Int64 = my.math::add(2, 3);
    print_int(sum);
}
```

- `@module` задаёт имя модуля; entry-файл не префиксуется
- `@pub` экспортирует символ; `@use module;` подключает модуль
- Имена других модулей префиксуются именем модуля (`add` → `my_math_add` при компиляции entry)
- Обращения к собственным функциям модуля внутри него остаются неквалифицированными

### Строки и коллекции

```glyph
let s: String = "Hello, " ++ "Glyph!";          // ++ — конкатенация
let upper: String = to_upper(s);
let n: Int64 = len(s);
let parsed: Option<Int64> = parse_int("42");    // Option<Int64>

let range: List<Int64> = 0..10;                 // диапазон-список [0..9]
let arr: List<Int64> = [1, 2, 3];               // литерал списка
print_int(arr.len());                           // 3 — длина
arr.append(4);                                  // рост буфера
for x in arr { total = total + x; }             // итерация по списку
let both: List<Int64> = arr ++ arr;             // конкатенация списков
let total: Int64 = 0;
for i in 1..=5 { total = total + i; }           // 15
```

### `Result` / `Option` с данными

`Result<T, E>` и `Option<T>` строятся через варианты и сопоставляются в `match`
с payload:

```glyph
@fn divide(a: Int64, b: Int64) -> Result<Int64, String> {
    if b == 0 {
        return Result::Err("division by zero");
    }
    return Result::Ok(a / b);
}

@fn main() {
    let q: Int64 = match divide(10, 2) {
        | Ok(v) => v,
        | Err(e) => -1
    };
    print_int(q);

    let p: Option<Float64> = parse_float("2.5");   // Some(payload) / None
    let f: Float64 = match p {
        | Some(v) => v,
        | None => 0.0
    };
    print_float(f);
}
```

### Обобщённые функции

Параметры типа объявляются после имени (`<T, K>`) и выводятся из аргументов
вызова; параметры, встречающиеся только в возвращаемом типе, выводятся из
аннотации `let`. Каждая инстанциация компилируется в отдельную C-функцию
(`identity_i64`, `identity_str`, ...):

```glyph
@fn identity<T>(x: T) -> T {
    return x;
}

@fn first<T, K>(a: T, b: K) -> T {
    return a;
}

@fn ok_wrap<T, E>(x: T) -> Result<T, E> {
    return Result::Ok(x);
}

@fn main() {
    let a: Int64 = identity(5);              // identity_i64
    let s: String = first("hello", 42);      // first_str_i64
    let r: Result<String, Int64> = ok_wrap("fine");  // E из аннотации
    print_int(a);
}
```

### Конкурентность

Вызов `@fn async` возвращает ленивый хендл `Async<T>` (поток не запущен).
`spawn` запускает хендл на отдельном потоке (pthreads), `await` ждёт результат
(выполняет синхронно, если не запущен; повторный `await` берёт кэш):

```glyph
@fn async fetch(url: String) -> String {
    return url;
}

@fn main() {
    let h: Async<String> = fetch("http://x");
    spawn h;
    println(h await);
}
```

Типизированные каналы `Channel<T>(capacity)`: ёмкость `0` — rendezvous
(прямая передача), `> 0` — буфер. `send` блокирует при полном буфере и
возвращает `false` на закрытом канале; `recv` блокирует при пустом и даёт
`None`, когда канал закрыт и пуст:

```glyph
@fn async produce(ch: Channel<Int64>) -> Int64 {
    ch.send(10);
    ch.send(20);
    ch.close();
    return 2;
}

@fn main() {
    let ch: Channel<Int64> = Channel<Int64>(4);
    spawn produce(ch);
    let m: Option<Int64> = ch.recv();
    let v: Int64 = match m {
        | Some(x) => x,
        | None => -1
    };
    print_int(v);
}
```

## Ассерты и тесты

```glyph
@module examples.calc

@fn add(a: Int64, b: Int64) -> Int64 { return a + b; }

@test @fn test_add() -> Void {
    assert_eq(add(2, 3), 5, "2 + 3 == 5");
    assert(add(2, 3) > 0, "positive result");
}
```

```bash
glyphc test        # сканирует ./src, собирает и запускает каждый файл с @test
glyphc test -i src/calc.glyph
```

Отчёт: `[ok]`/`[FAILED]` по каждому тесту, сводка `Result: OK|FAILED (N file(s), X passed, Y failed)`,
exit-код 1 при падении. Подробнее — [docs/testing.md](docs/testing.md).

## Стандартная библиотека

Библиотека встроена в компилятор (без `@use`). Исходники — в `stdlib/`.

| Функции | Описание |
|---|---|
| `print`, `println`, `eprintln`, `print_int`, `print_float`, `print_bool`, `read_line` | ввод/вывод |
| `sqrt`, `pow`, `abs`, `floor`, `ceil`, `round`, `min`, `max`, `clamp`, `log`, `log2`, `log10`, `sin`, `cos`, `tan` | математика |
| `len`, `substring`, `contains`, `starts_with`, `ends_with`, `replace`, `split`, `trim`, `to_upper`, `to_lower`, `char_at`, `parse_int`, `parse_float`, `int_to_string`, `float_to_string`, `bool_to_string` | строки |
| `file_read`, `file_write`, `file_append`, `file_exists`, `create_dir`, `remove_file`, `list_dir`, `file_copy`, `file_rename` | файлы |
| `to_int`, `to_float`, `to_bool` | конвертация |
| `alloc`, `free`, `memcpy`, `memset` | сырая память |
| `assert`, `assert_eq`, `assert_ne`, `assert_true`, `assert_false` | ассерты |

## Манифест проекта (glyph.toml)

Для `glyphc build` в текущей директории должен лежать `glyph.toml`:

```toml
name = "my-project"
compiler = "gcc"
optimization = "-O2"
```

`build` компилирует все `.glyph` из `src/` и линкует их в `build/main`.
`--profile release` использует `-O3`. В v1.0 `build` не запускает тесты — для этого есть `glyphc test`.

## Примеры

Все примеры в `examples/` — 22 файла + многофайловые проекты `examples/project/` и
`examples/tests/`. Проверить сразу всё:

```bash
examples/run_all.sh
```

| Пример | Что показывает |
|---|---|
| `hello.glyph` | вывод, строки, `++` |
| `simple.glyph` | минимум |
| `structs.glyph`, `impl_const.glyph` | структуры, свободные функции, `@const` |
| `enums.glyph`, `enum_test.glyph`, `enum_data.glyph`, `enum_payload.glyph` | перечисления, match, данные вариантов |
| `match.glyph` | match с паттернами |
| `lists.glyph` | списки: литералы, срезы, `==`, `.free()`, итерация |
| `list_refs.glyph` | refcount списков: `free()` с алиасами, copy-on-write при `append` |
| `fifteen.glyph` | интерактивные «пятнашки»: `read_line`, `List`, `@test`, LCG |
| `maps.glyph` | `Map<String, V>`: литерал `#{...}`, индекс, `get`/`put`/`len`/`free`, итерация `for k in m` |
| `while.glyph`, `v1_features.glyph` | циклы, касты, диапазоны, литералы списков |
| `math_test.glyph`, `stdlib_test.glyph` | std.math / std.io / std.string |
| `error_handling.glyph` | `#guard` + enum-ошибка |
| `generics.glyph` | обобщённые функции: инференс, `Option<T>`/`Result<T, E>`, цепочки вызовов |
| `concurrency.glyph` | `@fn async`, `spawn`/`await`, `Channel<T>`, `send`/`recv`/`close` |
| `project/` | модульная программа: `@use math`, `@use geometry` |
| `tests/` | тестовый проект: `@test`, ассерты, `glyphc test` |

## Структура репозитория

```
glyphc/
  src/            компилятор: lexer, parser, typechecker, codegen, modules, cli, lsp
  stdlib/         стандартная библиотека (исходники Glyph)
  examples/       рабочие примеры + run_all.sh
```

## Производительность

Glyph компилируется в C и наследует C-тулчейн: скалярный код (арифметика,
циклы) — в паритете с C, накладные расходы появлются только там, где вы
используете аллокации/управление памятью. Таблица — одинаковые алгоритмы на
одной машине (min из 3 прогонов, gcc 16.2.1 `-O2`, rustc 1.98.1 `--release`
+LTO; порядок величин, не строгий бенчмарк).

```
workload        glyph      c      rust   ratio (glyph/c)
loop_sum         468ms   466ms   541ms        1.00x
list_append      267ms    62ms    59ms        4.31x
map_put_get      346ms   152ms   169ms        2.28x
```

Железо: 11th Gen Intel Core i5-1135G7, 2026-09-08. `loop_sum` суммирует
`i % 7` по runtime-размеру (400M итераций); `list_append` — 20M int64 в
динамический массив и суммирование; `map_put_get` — 2M put+get строковых ключей
(разброс по 100 ключам). Воспроизводится `bench/gen.sh` + `bench/run.sh`.

- `loop_sum` — паритет с C (движок цикла/арифметика идентичны gcc; 541ms Rust —
  это cost-model LLVM на `%`)
- `list_append` платит за refcount и resize-копирование `GlyphList`
  (безопасный growable-список с runtime-проверками): ~4x к ручному realloc
- `map_put_get` — ~2.3x к C-мапе того же размера: хэширование строк + коробки
  `Option`-payload при каждой операции; будущая арена-мапа сократит разрыв

## Ограничения

- `Result`/`Option`: коробка `void*` (payload копируется в кучу); временные боксы
  (результат вызова в позиции scrutinee `match`) освобождаются автоматически,
  именованные — через `drop(box)`; payload-указатели остаются чужими
- Обобщения: только функции (без generic-структур/enum/impl, без trait bounds);
  вызов generic-функции внутри generic-тела требует конкретных типов
- `List<T>`: refcount (копии делят буфер, `free()` отпускает ref, `append` при
  нескольких владельцах — copy-on-write); `==`/`!=` — только POD-скаляры
  (`Int64`/`UInt64`/`Float64`/`Bool`, побайтово), структуры и списки указателей
  отклоняются громко; срезы `xs[a..b]`/`xs[a..=b]` — копия
- `Map<String, V>`: рантайм есть (`#{...}`, индекс, `put`/`get`/`len`/`free`),
  итерация `for k in m` даёт ключи по bucket-цепям (порядок = хэшу, не вставке);
  ключи — только `String`
- Конкурентность: без GC — хендлы/каналы не освобождаются (списки — через
  `xs.free()`); нет `select`, таймаутов и async-generic функций; `send(&ref)`
  запрещён (адрес стека)
- Ошибки компилятора: лексер/парсер — с `line:col`, тайпчекер — с контекстом
  функции (`in function 'foo': ...`); полных спанов AST пока нет
- LSP-сервер — экспериментальный

## Лицензия

MIT