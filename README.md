# Glyph Language Compiler (glyphc)

Компилятор языка программирования **Glyph** v1.0.0, написанный на Rust.

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
- Типизированные списки `List<T>`: литералы `[..]`, индексация `arr[i]`, диапазоны `0..10`
- `Result`/`Option` с данными: `Result::Ok(x)`/`Option::Some(x)` — построение и
  сопоставление (`Ok(v)`/`Some(v)` в `match`) с payload, в т.ч. из `parse_int`/`parse_float`
- Обобщённые функции: `@fn identity<T>(x: T) -> T` — мономорфизация, вывод типов
  по аргументам и из аннотации `let`, вложенные типы (`Option<T>`, `List<T>`)
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

let range: List<Int64> = 0..10;                 // диапазон-список
let arr: List<Int64> = [1, 2, 3];               // литерал списка
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

Все примеры в `examples/` — 16 файлов + многофайловый проект `examples/project/` + тестовый
проект `examples/tests/`. Проверить сразу всё:

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
| `while.glyph`, `v1_features.glyph` | циклы, касты, диапазоны, литералы списков |
| `math_test.glyph`, `stdlib_test.glyph` | std.math / std.io / std.string |
| `error_handling.glyph` | `#guard` + enum-ошибка |
| `generics.glyph` | обобщённые функции: инференс, `Option<T>`/`Result<T, E>`, цепочки вызовов |
| `project/` | модульная программа: `@use math`, `@use geometry` |
| `tests/` | тестовый проект: `@test`, ассерты, `glyphc test` |

## Структура репозитория

```
glyphc/
  src/            компилятор: lexer, parser, typechecker, codegen, modules, cli, lsp
  stdlib/         стандартная библиотека (исходники Glyph)
  examples/       рабочие примеры + run_all.sh
```

## Ограничения

- `Result`/`Option`: фиксированная коробка `void*` (payload копируется в кучу)
- Обобщения: только функции (без generic-структур/enum/impl, без trait bounds);
  вызов generic-функции внутри generic-тела требует конкретных типов
- `List<T>` динамические операции (`append`, `len`, рост размера) в работе;
  сейчас доступны литералы и индексация фиксированных списков
- Конкурентность (`spawn`, `await`, каналы) в поверхностном языке ещё нет
- LSP-сервер — экспериментальный
- Планы v1.1: динамические `List<T>`, конкурентность

## Лицензия

MIT