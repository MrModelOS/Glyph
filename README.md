# Glyph Language Compiler (glyphc)

Компилятор языка программирования Glyph v0.1.

## Установка

```bash
cargo build --release
```

## Использование

### Компиляция

```bash
glyphc compile --input file.glyph --output file.c
```

### Проверка синтаксиса

```bash
glyphc check --input file.glyph
```

### Компиляция и запуск

```bash
glyphc run --input file.glyph
```

### Сборка проекта

```bash
glyphc build
```

### Просмотр токенов

```bash
glyphc tokens --input file.glyph
```

### Просмотр AST

```bash
glyphc ast --input file.glyph
```

## Синтаксис языка

### Модули

```glyph
@module my.module
```

### Функции

```glyph
@fn add(a: Int64, b: Int64) -> Int64 {
    return a + b;
}
```

### Структуры

```glyph
@struct Point {
    x: Float64,
    y: Float64,
}
```

### Перечисления

```glyph
@enum Color {
    Red,
    Green,
    Blue,
}
```

### Контракты

```glyph
@fn divide(a: Float64, b: Float64) -> Result<Float64, Error> {
    #guard(b != 0.0) else {
        return Err(Error.DivisionByZero);
    };
    return Ok(a / b);
}
```

## Манифест проекта (glyph.toml)

```toml
[package]
name = "my-project"
version = "0.1.0"

[build]
compiler = "gcc"
optimization = "-O2"
```

## Стандартная библиотека

- `std.math` - математические функции
- `std.io` - ввод/вывод
- `std.string` - работа со строками

## LSP сервер

```bash
glyphc --lsp
```

## Примеры

Смотрите директорию `examples/`.

## Лицензия

MIT
