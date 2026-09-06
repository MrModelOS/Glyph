# Тестирование в Glyph

Атрибут `@test` + встроенные ассерты + подкоманда `glyphc test`.

## Атрибут `@test`

Любая функция может быть тестом:

```glyph
@test @fn test_add() -> Void {
    // ...
}
```

Правила:
- функция не принимает параметров;
- возвращает `Void`;
- грамматика `@test @fn` — любой порядок с `@pub` (`@pub @test @fn`, `@test @pub @fn`).

Нарушение проверяется на этапе типизации:

```
Type error: Test function 'test_with_param' must not have parameters
Type error: Test function 'test_bad_ret' must return Void (found Int64)
```

## Ассерты

Доступны без `@use` (аналоги в `stdlib/assert.glyph`):

```glyph
assert(condition: Bool, msg: String) -> Void
assert_true(condition: Bool, msg: String) -> Void
assert_false(condition: Bool, msg: String) -> Void
assert_eq(a, b, msg: String) -> Void   // a == b
assert_ne(a, b, msg: String) -> Void   // a != b
```

`assert_eq`/`assert_ne` полиморфны по типам: оба аргумента должны быть одного
типа из `Int64`, `UInt64`, `Float64`, `Bool`, `String`. При провале выводится `msg`.

В обычной (не тестовой) сборке проваленный ассерт аварийно завершает программу.

## glyphc test

Как это работает:

```bash
glyphc test                      # сканирует ./src рекурсивно, ищет *.glyph с @test
glyphc test -i src/calc.glyph    # только один файл
glyphc test --compiler clang     # сборщик
glyphc test --opt -O0            # уровень оптимизации
```

Для каждого файла:
1. ищутся все `@test`-функции (файлы без тестов пропускаются);
2. файл компилируется в обычном режиме + генерируется специальный C-рантайм,
   который выполняет каждый тест в изоляции (`setjmp`/`longjmp`) — падение одного
   теста не прерывает остальные;
3. построенная программа возвращает протокол в stdout.

Формат протокола от `glyphc test`:

```
Running 8 test(s) in src/calc.glyph...
  [ok] test_add
  [ok] test_add_negative
  [FAILED] test_fail
    this test should fail

Result: FAILED  (2 file(s), 9 passed, 1 failed)
```

- exit code `0` — все тесты прошли;
- exit code `1` — есть упавшие тесты (берётся программа, собравшаяся успешно);
- ошибки компиляции/типов по конкретному файлу не прерывают остальные файлы,
  результат суммируется.

## Пример проекта

`examples/tests/` — законченный тестовый проект из двух модулей:

```
examples/tests/
  glyph.toml
  src/calc.glyph    @module calc; 8 тестов (7 успешных + 1 намеренно падающий)
  src/main.glyph    @use calc; 2 теста, проверяют вызовы через calc::add
```

Запуск:

```bash
cd examples/tests
glyphc test
# Result: FAILED  (2 file(s), 9 passed, 1 failed)   — test_fail упал намеренно
glyphc test -i src/main.glyph
# Result: OK  (1 file(s), 2 passed, 0 failed)
```

## Регрессия компилятора

Юнит-тесты самого компилятора:

```bash
cargo test
```

Полный прогон всех примеров языка:

```bash
examples/run_all.sh
```