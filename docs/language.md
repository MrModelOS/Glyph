# Язык Glyph — справочник

Статус: v1.3.0. Документ описывает реально работающий синтаксис
(проверено на `examples/` и тестах). Незадокументированные конструкции не считаются частью языка.

## 1. Структура файла

Файл — последовательность объявлений верхнего уровня:

```glyph
@module examples.demo        // имя модуля верхнего уровня
@use other.module;            // импорт другого модуля

@const SCALE: Float64 = 1.5;

@struct Point { x: Float64, y: Float64 }

@enum Op { Plus, Minus, MulDiv(Float64, Float64) }

@fn twice(x: Int64) -> Int64 { return x * 2; }

@test @fn test_twice() -> Void { assert_eq(twice(2), 4, "2 * 2 == 4"); }
```

Атрибуты (пишутся через `@`):

| Атрибут | Применяется к | Назначение |
|---|---|---|
| `@module` | файл | имя модуля |
| `@use` | файл | импорт модуля |
| `@fn` | функция | объявление функции |
| `@struct` | структура | объявление структуры |
| `@enum` | перечисление | объявление перечисления |
| `@const` | константа | именованная константа |
| `@pub` | fn/struct/enum | публичная видимость (для импортируемых модулей) |
| `@test` | fn | тестовая функция (см. docs/testing.md) |

Комментарии: `// строка`, `/* блок */`.

## 2. Типы

| Тип | C-представление | Примечание |
|---|---|---|
| `Int64` | `int64_t` | целые |
| `UInt64` | `uint64_t` | беззнаковые |
| `Float64` | `double` | числа с плавающей точкой |
| `Bool` | `int` | `true`/`false` |
| `String` | `char*` | строки |
| `Bytes` | `char*` | сырые байты |
| `List<T>` | `GlyphList` (refcounted fat-struct) | литерал `[..]`, диапазон `..`, `.len()`, `.append()`, `.free()`, срезы, `for x in xs`, `++`, `==` для POD |
| `Option<T>` | `GlyphBox*` (в C — `void*`) | коробка `{tag, data}`; `Some(x)`/`None` |
| `Result<T, E>` | `GlyphBox*` (в C — `void*`) | коробка `{tag, data}`; `Ok(x)`/`Err(e)` |
| `&T` | `T*` | ссылка на значение (для перечислений в `match`) |
| `MyStruct`, `MyEnum` | struct/tagged union | пользовательские типы |

### `Result` / `Option` с данными

`Result<Ok, E>` и `Option<T>` строятся вариантами и сопоставляются в `match`.
Payload копируется в кучу (коробка `{ tag, data }`); тег `1` — `Ok`/`Some`,
`0` — `Err`/`None`:

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

    let parsed: Option<Float64> = parse_float("2.5");
    let f: Float64 = match parsed {
        | Some(v) => v,
        | None => 0.0
    };
    print_float(f);
}
```

Замечания:
- `Option::None`/`Result::Ok(x)` без контекста дают неизвестный параметр (`Type::Void`);
  в `let` он унифицируется с объявленным типом (`let n: Option<Int64> = Option::None`)
- `parse_int`/`parse_float`/`to_int`/`to_float` возвращают коробку `Option`
  (`Some`/`None`) и пригодны для `match`

### Обобщённые функции

Параметры типа объявляются в угловых скобках после имени функции и могут
использоваться в сигнатуре и теле, в том числе вложенно (`List<T>`,
`Option<T>`, `Result<T, E>`). Компилятор мономорфизирует каждую инстанциацию
в отдельную C-функцию (`identity_i64`, `first_str_i64`):

```glyph
@fn identity<T>(x: T) -> T {
    return x;
}

@fn first<T, K>(a: T, b: K) -> T {
    return a;
}

@fn main() {
    let a: Int64 = identity(5);          // T = Int64 из аргумента
    let s: String = first("hi", 42);     // T = String, K = Int64
    print_int(a);
}
```

Вывод типов:
- по типам аргументов вызова (включая вложенные: `List<Int64>` даёт `T = Int64`);
- из аннотации `let` для параметров, встречающихся только в возвращаемом типе
  (`let r: Result<String, Int64> = ok_wrap("fine")` выводит `E = Int64`).

Ограничения v1.3: обобщаются только функции (структуры/enum/impl —
конкретные); явные аргументы типов (`f::<Int64>`) не поддерживаются;
вызов generic-функции внутри generic-тела требует конкретных типов
(иначе — ошибка `Cannot infer type arguments`); итерация по `Map` проверена,
но не реализована (рантайм есть, `for k in m` отвергается на этапе проверки
типов); тип `[T; N]` описан, но литерал всегда порождает `List<T>`, поэтому
массивы неконструируемы.

Литералы:

```glyph
42         // Int64
498_351_8   // разделители разрядов
3.14       // Float64
true false // Bool
"hello"    // String
0xFF       // hex
```

## 3. Переменные и константы

```glyph
let x: Int64 = 42;            // неизменяемая
let mut acc: Float64 = 0.0;   // изменяемая
let y = 42;                   // untyped let: тип выводится (Int64, String, List<T>, ...)
let r = 0..5;                 // List<Int64]
let s = "a" ++ "b";           // String

x = 10;        // ❌ ошибка типов: x неизменяемая
acc = acc + 1; // ✅

@const PI: Float64 = 3.141592653589793;
@const MAX_RETRIES: Int64 = 3;
```

Untyped `let` выводит тип из значения (литералы, `List`, `Range`,
конкатенация, вызовы). Неизвестный тип — ошибка кодгена
(`cannot infer type of untyped let`), а не мусор в C.

Константы компилируются в `static const`-выражения и могут ссылаться
на другие константы и литералы.

## 4. Функции

```glyph
@fn add(a: Int64, b: Int64) -> Int64 { return a + b; }

@fn log_message(msg: String) -> Void { println(msg); }

@fn default() -> Int64 { 0 }   // тело-выражение
@fn make() -> Float64 { return 2.0; }
```

- Параметры типизированы; тип возврата указывается через `->`, `Void` — без значения.
- Аргументы передаются по значению; доступны ссылки `&T`.
- Неквалифицированные вызовы внутри собственного модуля работают как есть.
- `@fn async` (и `@fn async` методы в `@impl`) возвращают ленивый хендл —
  см. раздел «Конкурентность» ниже. Async-generic функции не поддерживаются.

## 5. Операторы

Арифметика: `+ - * / %`.

Сравнения: `== != < > <= >=`. Логика: `&& || !`.

Конкатенация строк: `++` — `"Hello, " ++ "World!"`. При конкатенации
с числами используйте конвертацию: `"x=" ++ int_to_string(x)`.

Приведение типов `as`: `let f: Float64 = 42 as Float64;`.

Ссылка `&`: `let r: &Shape = &circle;` (используется для `match` по значению
без копирования).

## 6. Управляющий поток

### if / else

```glyph
if x > 0 {
    print("positive");
} else if x < 0 {
    print("negative");
} else {
    print("zero");
}
```

### while

```glyph
let i: Int64 = 0;
while i < 5 {
    i = i + 1;
}
```

### loop / break / continue

```glyph
let i: Int64 = 0;
loop {
    i = i + 1;
    if i >= 100 { break; }
}
```

### for .. in (целочисленный диапазон)

```glyph
let sum: Int64 = 0;
for i in 0..5 {        // 0..5 — не включая 5 → сумма 10
    sum = sum + i;
}
for j in 1..=5 {       // 1..=5 — включительно
    sum = sum + j;
}
```

### match

```glyph
match value {
    | 1 => print("one")
    | 2 => print("two")
    | _ => print("other")
}

match status {
    | HttpStatus::Ok => "OK"
    | HttpStatus::NotFound => "Not Found"
    | _ => "Unknown"
}

match shape {
    | Shape::Circle(r) => 3.14159 * r * r
    | Shape::Rectangle(w, h) => w * h
    | Shape::Point => 0.0
}
```

Паттерны: литералы, `Enum::Variant`, `Enum::Variant(bindings)`, `_` (wildcard).
Body arms — выражения (в т.ч. вызовы, завершившиеся `;`).

### guard

```glyph
@fn divide(a: Float64, b: Float64) -> DivisionResult {
    #guard(b != 0.0) else {
        return DivisionResult::DivisionByZero;
    };
    return DivisionResult::Success(a / b);
}
```

`#guard(условие) else { ... };` — если условие ложно, выполняется блок `else`
и текущая функция возвращается; после `else` обязательна `;`.

## 7. Строки и списки

```glyph
let greeting: String = "Hello, " ++ "Glyph!";
let n: Int64 = len(greeting);                // 13
let sub: String = substring(greeting, 0, 5); // "Hello"
let has: Bool = contains(greeting, "Glyph"); // true
let upper: String = to_upper(greeting);      // "HELLO, GLYPH!"
let words: String = split("a,b,c", ",");     // CSV-фрагмент
let num: Option<Int64> = parse_int("42");    // Option<Int64>
let s: String = int_to_string(42);

let arr: List<Int64> = [10, 20, 30];          // литерал списка
print_int(arr[1]);                            // 20 — индексация
let points: List<P> = [P { x: 1.0, y: 2.0 }];
print_float(points[0].y);                     // 2.0 — индексация структур
let range: List<Int64> = 0..10;   // [0, 1, ..., 9]
print_int(range.len());           // 10 — длина
range.append(10);                 // рост буфера
let both: List<Int64> = arr ++ range;  // конкатенация списков
for x in arr {                    // итерация по списку
    print_int(x);
}
```

Списки — fat-struct `{data, len, elem_size, refs}` поверх heap-буфера:
литералы и диапазоны выделяют память, `append` растягивает буфер, индексация
`arr[i]` возвращает элемент `T`. Копирование списка (`let b = a`, передача в
функцию) считается ref-count'ом: буфер жив, пока жива хотя бы одна копия,
поэтому `a.free(); b[0]` безопасно. `append` при нескольких владельцах делает
copy-on-write (отдельный буфер для этого владельца; `append` внутри функции
не виден снаружи). Тип элемента литерала выводится по первому элементу.
`let` без аннотации тоже выводит тип (`let xs = [1, 2]` даёт `List<Int64>`).

Срезы `xs[a..b]` / `xs[a..=b]` возвращают новый `List<T>` (копия диапазона,
границы клампятся). Списки POD-скаляров (`Int64`, `UInt64`, `Float64`,
`Bool`) сравниваются по значению (`==`/`!=` через `len` + `memcmp`);
остальные — громкая ошибка кодгена. `xs.free()` отпускает один ref и
обнуляет список (буфер остаётся, пока есть другие копии).

### 7.1. Map

`Map<String, V>` — chained-хэшмап (FNV-1a, ключ всегда `String`), значения
копируются по `elem_size(V)`. Литерал `#{ "key": value, ... }` строит map;
тип значения выводится по первой паре (пустой — `Map<String, Void>`).

```glyph
@fn main() -> Void {
    let m: Map<String, Int64> = #{ "alice": 90, "bob": 75 };
    print_int(m["alice"]);          // чтение; отсутствующий ключ — abort
    m["carol"] = 60;                // запись через индекс
    m.put("bob", 80);               // запись методом
    let got: Option<Int64> = m.get("dave");   // Option вместо abort
    let v: Int64 = match got {
        | Some(x) => x
        | None => 0
    };
    drop(got);                      // Option-бокс требует drop(box)
    print_int(m.len());
    m.free();                       // освободить хранилище
}
```

Внимание: `m["k"]` по отсутствующему ключу аварийно завершает программу —
для проверяемого доступа используйте `m.get(k)` (возвращает `Option<V>`;
`Some`-бокс освобождается через `drop(box)`). Итерация `for k in m` не
реализована и отвергается на этапе проверки типов.

## 8. Структуры и методы

```glyph
@struct Point {
    x: Float64,
    y: Float64,
}

let p: Point = Point { x: 0.0, y: 0.0 };
let dx: Float64 = p2.x - p1.x;
```

### Методы через `@impl`

`@impl <Type> { @fn ... }` объявляет методы типа. Ресивер передаётся
**первым параметром**, вызов `obj.method(a, b)` превращается в
`Type_method(obj, a, b)`.

```glyph
@impl Point {
    @fn norm(p: Point) -> Float64 {
        return sqrt(p.x * p.x + p.y * p.y);
    }
    @fn scaled(p: Point, k: Float64) -> Point {
        return Point { x: p.x * k, y: p.y * k };
    }
}

let p: Point = Point { x: 3.0, y: 4.0 };
let n: Float64 = p.norm();          // → Point_norm(p)
let q: Point = p.scaled(2.0);       // → Point_scaled(p, 2.0)
```

Правила:
- первый параметр метода должен иметь тип (или `&`-ссылку на тип) `Type` —
  тот же, что и у объекта вызова;
- остальные параметры — обычные аргументы; `args.len() == params.len() - 1`;
- тело метода обращается к полям ресивера через параметр (`p.x`), без `self`;
- методы других модулей не префиксуются (§10), имена методов —
  `Type_method`; методы нельзя импортировать как функции.

## 9. Перечисления

```glyph
@enum Color { Red, Green, Blue }              // unit-варианты

@enum Shape {
    Circle(Float64),                          // вариант с данными
    Rectangle(Float64, Float64),
    Point,
}

@enum HttpStatus {                            // дискриминанты
    Ok = 200,
    NotFound = 404,
}
```

Конструкция: `Shape::Circle(5.0)`, `HttpStatus::Ok`, `Color::Red`.

Сопоставление:

```glyph
match s {
    | Shape::Circle(r) => 3.14159 * r * r
    | Shape::Rectangle(w, h) => w * h
    | Shape::Point => 0.0
}
```

**Правило разбора `::`**: отрезот `патh` с заглавной буквы — конструктор перечисления
(`Shape::Circle(...)`); со строчной — квалифицированный вызов функции модуля (`math::add(...)`).

## 10. Модули

```glyph
// math.glyph  —— модуль
@module my.math
@pub @fn add(a: Int64, b: Int64) -> Int64 { return a + b; }

// main.glyph —— entry
@use my.math;
@fn main() -> Void {
    let sum: Int64 = my.math::add(1, 2);
}
```

- Entry-файл (тот, что передаётся в `glyphc run/compile`) не префиксуется.
- Имена импортированных модулей префиксуются (`add` → `my_math_add`); ссылки
  на собственные функции внутри модуля переписываются автоматически.
- `@pub` — публичность; по умолчанию символы приватные, но в v1.0 маркировка
  не является жёстким ограничением.

## 11. Тесты и ассерты

См. [docs/testing.md](testing.md).

## 12. Конкурентность

Вызов `@fn async` не выполняет тело, а возвращает ленивый хендл `Async<T>`.
`spawn h;` запускает хендл на отдельном потоке (pthreads, идемпотентно),
`h await` ждёт результат: выполняет синхронно, если поток не запущен, иначе
join; повторный `await` возвращает кэшированное значение:

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

Типизированные каналы создаются конструктором `Channel<T>(capacity)`:
ёмкость `0` — rendezvous (прямая передача без буфера), `> 0` — буфер.
`send(v)` блокирует при полном буфере и возвращает `false`, если канал закрыт;
`recv()` блокирует при пустом буфере и возвращает `None`, когда канал закрыт
и пуст; `close()` будит всех ожидающих:

```glyph
@fn async produce(ch: Channel<Int64>) -> Int64 {
    ch.send(1);
    ch.send(2);
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

Ограничения: рантайм без GC (хендлы, каналы и списки не освобождаются);
`Option`/`Result`-боксы: временные (результат вызова прямо в `match`)
освобождаются автоматически, именованные — через `drop(box)`; нет
`select`/таймаутов; async-generic функции запрещены; `send(&ref)` запрещён
типом (адрес стека нельзя передавать между потоками).

## 13. Стандартная библиотека

Библиотека доступна без `@use`; сигнатуры объявлены в компиляторе,
тела — в `stdlib/`.

```glyph
print(msg: String) -> Void
println(msg: String) -> Void
eprintln(msg: String) -> Void
print_int(value: Int64) -> Void
print_float(value: Float64) -> Void
print_bool(value: Bool) -> Void
read_line() -> String

sqrt(x: Float64) -> Float64
pow(base: Float64, exp: Float64) -> Float64
abs(x: Float64) -> Float64
floor(x: Float64) -> Int64
ceil(x: Float64) -> Int64
round(x: Float64) -> Int64
min(a: Float64, b: Float64) -> Float64
max(a: Float64, b: Float64) -> Float64
clamp(value: Float64, low: Float64, high: Float64) -> Float64
log(x: Float64) -> Float64      log2(x: Float64) -> Float64
log10(x: Float64) -> Float64    sin(x: Float64) -> Float64
cos(x: Float64) -> Float64      tan(x: Float64) -> Float64

len(s: String) -> Int64
substring(s: String, start: Int64, end: Int64) -> String
contains(s: String, sub: String) -> Bool
starts_with(s: String, prefix: String) -> Bool
ends_with(s: String, suffix: String) -> Bool
replace(s: String, from: String, to: String) -> String
split(s: String, delimiter: String) -> String
trim(s: String) -> String
to_upper(s: String) -> String    to_lower(s: String) -> String
char_at(s: String, index: Int64) -> String
parse_int(s: String) -> Option<Int64>
parse_float(s: String) -> Option<Float64>
int_to_string(value: Int64) -> String
float_to_string(value: Float64) -> String
bool_to_string(value: Bool) -> String

to_int(s: String) -> Option<Int64>
to_float(s: String) -> Option<Float64>
to_bool(s: String) -> Option<Bool>

file_read(path: String) -> String
file_write(path: String, content: String) -> Bool
file_append(path: String, content: String) -> Bool
file_exists(path: String) -> Bool
create_dir(path: String) -> Bool
remove_file(path: String) -> Bool
list_dir(path: String) -> String
file_copy(src: String, dst: String) -> Bool
file_rename(old: String, new: String) -> Bool

alloc(size: Int64) -> Bytes
free(ptr: Bytes) -> Void
memcpy(dst: Bytes, src: Bytes, size: Int64) -> Void
memset(dst: Bytes, value: Int64, size: Int64) -> Void

assert(condition: Bool, msg: String) -> Void
assert_eq(a: Int64|UInt64|Float64|Bool|String, b: ..., msg: String) -> Void
assert_ne(a: ..., b: ..., msg: String) -> Void
assert_true(condition: Bool, msg: String) -> Void
assert_false(condition: Bool, msg: String) -> Void
```