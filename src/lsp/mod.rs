use std::collections::HashMap;
use std::io::{self, BufRead, Read, Write};

use serde_json::{json, Value};

use crate::ast::{LineCol, Program, Span, Stmt, TopLevelItem};
use crate::lexer::{Lexer, LexerError};
use crate::parser::{ParseError, Parser};
use crate::typechecker::{TypeChecker, TypeError};

#[derive(Debug)]
pub struct LspServer {
    documents: HashMap<String, Document>,
    running: bool,
}

#[derive(Debug)]
struct Document {
    content: String,
    program: Option<Program>,
}

impl LspServer {
    pub fn new() -> Self {
        LspServer {
            documents: HashMap::new(),
            running: true,
        }
    }

    pub fn run(&mut self) {
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut reader = stdin.lock();
        let mut writer = stdout.lock();

        while self.running {
            let mut content_length: Option<usize> = None;

            // Parse headers until the blank separator line.
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => return, // EOF
                    Ok(_) => {}
                    Err(_) => return,
                }
                let line = line.trim_end();
                if line.is_empty() {
                    break;
                }
                if let Some(rest) = line.strip_prefix("Content-Length:") {
                    content_length = rest.trim().parse::<usize>().ok();
                }
            }

            let Some(len) = content_length else {
                continue;
            };
            let mut body = vec![0u8; len];
            if reader.read_exact(&mut body).is_err() {
                return;
            }
            let Ok(json_str) = String::from_utf8(body) else {
                continue;
            };
            let Ok(msg) = serde_json::from_str::<Value>(&json_str) else {
                continue;
            };

            let responses = self.handle_message(&msg);
            for resp in responses {
                let bytes = resp.as_bytes();
                write!(writer, "Content-Length: {}\r\n\r\n{}", bytes.len(), resp).unwrap();
                writer.flush().unwrap();
            }
        }
    }

    fn handle_message(&mut self, msg: &Value) -> Vec<String> {
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let id = msg.get("id").cloned();

        match method {
            "initialize" => vec![
                json!({
                    "jsonrpc": "2.0",
                    "id": id.unwrap_or(Value::Null),
                    "result": {
                        "capabilities": {
                            "textDocumentSync": 1,
                            "completionProvider": {
                                "triggerCharacters": ["@", "#", ".", ":"]
                            },
                            "hoverProvider": true,
                            "definitionProvider": true,
                            "diagnosticProvider": {
                                "interFileDependencies": false,
                                "workspaceDiagnostics": false
                            }
                        }
                    }
                })
                .to_string(),
            ],
            "initialized" => Vec::new(),
            "shutdown" => vec![
                json!({
                    "jsonrpc": "2.0",
                    "id": id.unwrap_or(Value::Null),
                    "result": null
                })
                .to_string(),
            ],
            "exit" => {
                self.running = false;
                Vec::new()
            }
            "textDocument/didOpen" => {
                self.upsert_document(msg);
                self.publish_diagnostics(msg)
            }
            "textDocument/didChange" => {
                self.upsert_document(msg);
                self.publish_diagnostics(msg)
            }
            "textDocument/completion" => vec![
                json!({
                    "jsonrpc": "2.0",
                    "id": id.unwrap_or(Value::Null),
                    "result": self.handle_completion()
                })
                .to_string(),
            ],
            "textDocument/hover" => self.handle_hover(msg),
            "textDocument/definition" => self.handle_definition(msg),
            "textDocument/diagnostic" => {
                let items = self.collect_diagnostics(msg);
                vec![
                    json!({
                        "jsonrpc": "2.0",
                        "id": id.unwrap_or(Value::Null),
                        "result": {
                            "kind": "full",
                            "items": items
                        }
                    })
                    .to_string(),
                ]
            }
            _ => Vec::new(),
        }
    }

    /// Store the document content from a didOpen/didChange notification.
    fn upsert_document(&mut self, msg: &Value) {
        let Some(uri) = msg.pointer("/params/textDocument/uri").and_then(|u| u.as_str()) else {
            return;
        };
        let content = msg
            .pointer("/params/contentChanges/0/text")
            .and_then(|c| c.as_str())
            .or_else(|| msg.pointer("/params/textDocument/text").and_then(|t| t.as_str()))
            .unwrap_or("");
        let program = Lexer::new(content)
            .tokenize()
            .ok()
            .and_then(|tokens| Parser::new(tokens).parse_program().ok());
        self.documents
            .insert(uri.to_string(), Document { content: content.to_string(), program });
    }

    /// Compute diagnostics for every open document.
    fn collect_diagnostics(&mut self, msg: &Value) -> Vec<Value> {
        let mut items = Vec::new();
        let Some(uri) = msg.pointer("/params/textDocument/uri").and_then(|u| u.as_str()) else {
            return items;
        };
        if let Some(doc) = self.documents.get(uri) {
            for d in analyze(&doc.content) {
                items.push(d);
            }
        }
        items
    }

    fn publish_diagnostics(&mut self, msg: &Value) -> Vec<String> {
        let Some(uri) = msg.pointer("/params/textDocument/uri").and_then(|u| u.as_str()) else {
            return Vec::new();
        };
        let Some(doc) = self.documents.get(uri) else {
            return Vec::new();
        };
        let items: Vec<Value> = analyze(&doc.content);
        vec![
            json!({
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {
                    "uri": uri,
                    "diagnostics": items
                }
            })
            .to_string(),
        ]
    }

    fn handle_definition(&mut self, msg: &Value) -> Vec<String> {
        let Some(uri) = msg.pointer("/params/textDocument/uri").and_then(|u| u.as_str()) else {
            return Vec::new();
        };
        let line = msg.pointer("/params/position/line").and_then(|l| l.as_u64()).unwrap_or(0) as usize;
        let character = msg
            .pointer("/params/position/character")
            .and_then(|c| c.as_u64())
            .unwrap_or(0) as usize;
        let Some(doc) = self.documents.get(uri) else {
            return Vec::new();
        };
        let Some(word) = word_at(&doc.content, line, character) else {
            return Vec::new();
        };
        let Some(span) = resolve_definition(doc.program.as_ref(), (line, character), &word) else {
            return Vec::new();
        };

        vec![
            json!({
                "jsonrpc": "2.0",
                "id": msg.get("id").cloned().unwrap_or(Value::Null),
                "result": {
                    "uri": uri,
                    "range": range_from(span.start, span.end)
                }
            })
            .to_string(),
        ]
    }

    fn handle_completion(&self) -> Value {
        let keywords = [
            "@module", "@use", "@fn", "@async", "@struct", "@enum", "@trait", "@impl",
            "@const", "@pub", "@test", "let", "mut", "if", "else", "match", "loop", "while",
            "for", "in", "return", "break", "continue", "spawn", "await", "select", "timeout",
            "default", "as", "true", "false", "#guard", "#inject",
        ];
        let completions: Vec<Value> = keywords
            .iter()
            .map(|k| json!({ "label": k, "kind": 14, "detail": "Glyph keyword" }))
            .collect();
        json!(completions)
    }

    fn handle_hover(&mut self, msg: &Value) -> Vec<String> {
        let Some(uri) = msg.pointer("/params/textDocument/uri").and_then(|u| u.as_str()) else {
            return Vec::new();
        };
        let line = msg.pointer("/params/position/line").and_then(|l| l.as_u64()).unwrap_or(0) as usize;
        let character = msg
            .pointer("/params/position/character")
            .and_then(|c| c.as_u64())
            .unwrap_or(0) as usize;
        let Some(doc) = self.documents.get(uri) else {
            return Vec::new();
        };
        let Some(word) = word_at(&doc.content, line, character) else {
            return Vec::new();
        };
        let value = hover_text(&word);
        vec![
            json!({
                "jsonrpc": "2.0",
                "id": msg.get("id").cloned().unwrap_or(Value::Null),
                "result": {
                    "contents": {
                        "kind": "markdown",
                        "value": value
                    }
                }
            })
            .to_string(),
        ]
    }
}

/// The word (identifier/keyword run) under the given 0-based position.
fn word_at(content: &str, line: usize, character: usize) -> Option<String> {
    let src = content.lines().nth(line)?;
    let chars: Vec<char> = src.chars().collect();
    if character >= chars.len() {
        return None;
    }
    let is_word = |c: char| c.is_alphanumeric() || c == '_' || c == '#' || c == '@';
    let mut start = character;
    let mut end = character;
    while start > 0 && is_word(chars[start - 1]) {
        start -= 1;
    }
    while end < chars.len() && is_word(chars[end]) {
        end += 1;
    }
    if start == end {
        return None;
    }
    Some(chars[start..end].iter().collect())
}

fn hover_text(word: &str) -> String {
    match word {
        "select" => "**select** — waits for the first ready event: `| v: T <- ch.recv() => ...`, `| r: T <- h await => ...`, `| timeout(ms) => ...`, `| default => ...`.".to_string(),
        "spawn" => "**spawn** — schedules an async handle on the M:N worker pool.".to_string(),
        "await" => "**await** — waits for an `Async<T>` handle and yields its value.".to_string(),
        "timeout" => "**timeout(ms)** — a `select` arm that fires after `ms` milliseconds.".to_string(),
        "Channel" => "**Channel<T>(cap)** — buffered (`cap > 0`) or rendezvous (`cap == 0`) channel; `send`/`recv`/`close`.".to_string(),
        "Async" => "**Async<T>** — a lazily-evaluated handle to an `@fn async` call.".to_string(),
        "Result" => "**Result<T, E>** — `Result::Ok(v)` / `Result::Err(e)`; error paths use `?`-free `match`.".to_string(),
        "Option" => "**Option<T>** — `Option::Some(v)` / `Option::None()`. A `None` box must be released with `drop`.".to_string(),
        "String" => "**String** — runtime string (char buffer); concatenation with `++`.".to_string(),
        "List" => "**List<T>** — growable list; methods: `append`, `push`, `get`, `set`, `len`, `free`, iteration with `for`.".to_string(),
        "Map" => "**Map<K, V>** — hash map; `put`, `get`, `len`, `free`, index read/write, iteration yields keys.".to_string(),
        "@fn" => "**@fn** — function declaration; `@fn async` runs its body on the worker pool.".to_string(),
        _ => format!("`{word}` — Glyph keyword/identifier."),
    }
}

fn range_from(start: LineCol, end: LineCol) -> Value {
    json!({
        "start": { "line": start.line.saturating_sub(1), "character": start.col.saturating_sub(1) },
        "end": {
            "line": end.line.saturating_sub(1),
            "character": if end.line == start.line {
                end.col.max(start.col).saturating_sub(1)
            } else {
                end.col.saturating_sub(1)
            }
        }
    })
}

fn diagnostic(range: Value, message: String) -> Value {
    json!({
        "range": range,
        "severity": 1,
        "source": "glyphc",
        "message": message
    })
}

fn type_error_range(e: &TypeError) -> (LineCol, LineCol) {
    match e {
        TypeError::InFunction { source, .. } => type_error_range(source),
        TypeError::AtSpan { span: Span { start, end }, .. } => (*start, *end),
        TypeError::AtLine { loc, .. } => (*loc, *loc),
        _ => (LineCol { line: 1, col: 1 }, LineCol { line: 1, col: 1 }),
    }
}

fn type_error_message(e: &TypeError) -> String {
    match e {
        TypeError::InFunction { source, .. } => type_error_message(source),
        TypeError::AtSpan { source, .. } | TypeError::AtLine { source, .. } => type_error_message(source),
        other => format!("{}", other),
    }
}

fn parse_error_info(e: &ParseError) -> (String, LineCol) {
    match e {
        ParseError::UnexpectedToken(tok, line, col, expected) => (
            format!("Unexpected token {:?}: expected {}", tok, expected),
            LineCol { line: *line, col: *col },
        ),
        ParseError::LexerError(inner) => lexer_error_info(inner),
    }
}

fn lexer_error_info(e: &LexerError) -> (String, LineCol) {
    match e {
        LexerError::UnexpectedChar(ch, line, col) => (
            format!("Unexpected character '{}'", ch),
            LineCol { line: *line, col: *col },
        ),
        LexerError::UnterminatedString(line, col) => (
            "Unterminated string".to_string(),
            LineCol { line: *line, col: *col },
        ),
        LexerError::InvalidNumber(line, col) => (
            "Invalid number".to_string(),
            LineCol { line: *line, col: *col },
        ),
    }
}

/// Lex, parse and type-check a source buffer, producing LSP diagnostics.
/// Ranges use the full spans produced by the parser/typechecker.
fn analyze(content: &str) -> Vec<Value> {
    let mut out = Vec::new();

    let tokens = match Lexer::new(content).tokenize() {
        Ok(t) => t,
        Err(e) => {
            let (msg, loc) = lexer_error_info(&e);
            out.push(diagnostic(range_from(loc, loc), msg));
            return out;
        }
    };

    let program: Program = match Parser::new(tokens).parse_program() {
        Ok(p) => p,
        Err(e) => {
            let (msg, loc) = parse_error_info(&e);
            out.push(diagnostic(range_from(loc, loc), msg));
            return out;
        }
    };

    if let Err(e) = TypeChecker::new().check_program(&program) {
        let (start, end) = type_error_range(&e);
        out.push(diagnostic(range_from(start, end), type_error_message(&e)));
    }

    out
}

#[derive(Debug, Clone)]
struct Def {
    name: String,
    span: Span,
    item: usize,
    is_global: bool,
}

fn lc_le(a: &LineCol, b: &LineCol) -> bool {
    a.line < b.line || (a.line == b.line && a.col <= b.col)
}

fn lc_gt(a: &LineCol, b: &LineCol) -> bool {
    a.line > b.line || (a.line == b.line && a.col > b.col)
}

fn push_stmt_defs(defs: &mut Vec<Def>, item: usize, stmt: &Stmt) {
    match stmt {
        Stmt::Let { name, name_span, .. } => {
            defs.push(Def { name: name.clone(), span: *name_span, item, is_global: false });
        }
        Stmt::For { variable, var_span, body, .. } => {
            defs.push(Def { name: variable.clone(), span: *var_span, item, is_global: false });
            for s in body {
                push_stmt_defs(defs, item, s);
            }
        }
        Stmt::Loop(_, body)
        | Stmt::While { body, .. }
        | Stmt::Guard { else_body: body, .. } => {
            for s in body {
                push_stmt_defs(defs, item, s);
            }
        }
        Stmt::Select { arms, .. } => {
            for arm in arms {
                for s in &arm.body {
                    push_stmt_defs(defs, item, s);
                }
            }
        }
        _ => {}
    }
}

/// Collect, per top-level item, the definitions (function/struct names and
/// any lexically-bound local names) together with the item start positions.
fn collect_defs(program: &Program) -> (Vec<Def>, Vec<(usize, LineCol)>) {
    let mut defs = Vec::new();
    let mut items = Vec::new();

    for (i, item) in program.items.iter().enumerate() {
        match item {
            TopLevelItem::Function { name, name_span, params, body, .. } => {
                items.push((i, name_span.start));
                defs.push(Def { name: name.clone(), span: *name_span, item: i, is_global: true });
                for p in params {
                    defs.push(Def { name: p.name.clone(), span: p.name_span, item: i, is_global: false });
                }
                for s in body {
                    push_stmt_defs(&mut defs, i, s);
                }
            }
            TopLevelItem::Struct { name, name_span, .. } => {
                items.push((i, name_span.start));
                defs.push(Def { name: name.clone(), span: *name_span, item: i, is_global: true });
            }
            TopLevelItem::Impl { methods, .. } => {
                let start = methods
                    .iter()
                    .flat_map(|m| {
                        m.params
                            .iter()
                            .map(|p| p.name_span.start)
                            .chain(m.body.iter().map(|s| s.loc()))
                    })
                    .reduce(|acc, s| if lc_le(&s, &acc) { s } else { acc });
                if let Some(s) = start {
                    items.push((i, s));
                }
                for m in methods {
                    for p in &m.params {
                        defs.push(Def { name: p.name.clone(), span: p.name_span, item: i, is_global: false });
                    }
                    for s in &m.body {
                        push_stmt_defs(&mut defs, i, s);
                    }
                }
            }
            _ => {}
        }
    }

    (defs, items)
}

/// Resolve the definition reachable from the identifier under the cursor.
/// Local bindings (function params, lets) win if one is defined before the
/// cursor in the enclosing top-level item; otherwise a matching
/// function/struct name is returned.
fn resolve_definition(program: Option<&Program>, pos: (usize, usize), word: &str) -> Option<Span> {
    let program = program?;
    let (defs, items) = collect_defs(program);
    let pos_lc = LineCol { line: pos.0 + 1, col: pos.1 + 1 };

    let cur = items
        .iter()
        .filter(|(_, s)| lc_le(s, &pos_lc))
        .map(|(i, _)| *i)
        .max();

    let mut best: Option<Span> = None;
    for d in defs.iter().filter(|d| !d.is_global && d.name == word) {
        if let Some(ci) = cur {
            if d.item != ci {
                continue;
            }
        } else {
            continue;
        }
        if lc_le(&d.span.start, &pos_lc)
            && (best.is_none() || lc_gt(&d.span.start, &best.as_ref().unwrap().start))
        {
            best = Some(d.span);
        }
    }
    if let Some(b) = best {
        return Some(b);
    }

    defs.iter()
        .filter(|d| d.is_global && d.name == word)
        .map(|d| d.span)
        .next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::to_string;

    #[test]
    fn analyze_reports_parse_error_with_position() {
        let src = "@fn main() -> Void {\n    let x: Int64 = 1 +\n}\n";
        let diags = analyze(src);
        assert_eq!(diags.len(), 1, "expected one diagnostic");
        let d = &diags[0];
        assert_eq!(d["severity"], 1);
        let msg = d["message"].as_str().unwrap();
        assert!(msg.starts_with("Unexpected token"), "got {}", msg);
        assert_eq!(d["range"]["start"]["line"], 2);
    }

    #[test]
    fn analyze_reports_type_error_with_full_span() {
        let src = "@fn main() -> Void {\n    let mut n: Int64 = 0;\n    let x: Int64 = n + 1 + nope;\n}\n";
        let diags = analyze(src);
        assert_eq!(diags.len(), 1, "expected one diagnostic: {:?}", diags);
        let d = &diags[0];
        assert_eq!(d["message"], "Undefined variable: nope");
        // 0-based: line 3, `nope` occupies characters 27..31 (exclusive end).
        assert_eq!(d["range"]["start"]["line"], 2);
        assert_eq!(d["range"]["start"]["character"], 27);
        assert_eq!(d["range"]["end"]["character"], 31);
    }

    #[test]
    fn analyze_returns_empty_for_valid_program() {
        let src = "@fn main() -> Void {\n    let x: Int64 = 1 + 2;\n    print_int(x);\n}\n";
        assert!(analyze(src).is_empty(), "got {:?}", analyze(src));
    }

    #[test]
    fn diagnostic_message_is_json() {
        let src = "@fn main() -> Void {\n    1++\n}\n";
        let d = analyze(&format!("{}\n", src));
        assert!(to_string(&d).is_ok());
    }

    #[test]
    fn word_at_finds_identifier_under_cursor() {
        assert_eq!(word_at("let count: Int64 = 1;", 0, 6), Some("count".to_string()));
        assert_eq!(word_at("let count: Int64 = 1;", 0, 100), None);
        assert_eq!(word_at("select {", 0, 1), Some("select".to_string()));
    }

    #[test]
    fn definition_resolves_local_let_and_param() {
        let src = "@fn main() -> Void {\n    let x: Int64 = 5;\n    let y: Int64 = x + 1;\n}\n";
        let program = Lexer::new(src).tokenize().ok()
            .and_then(|t| Parser::new(t).parse_program().ok())
            .unwrap();
        // Cursor over the usage of `x` on line 3: resolves to the let on line 2.
        let span = resolve_definition(Some(&program), (2, 21), "x").unwrap();
        assert_eq!((span.start.line, span.start.col), (2, 9));
        // Param-style: use a function argument in the body.
        let src2 = "@fn add(a: Int64, b: Int64) -> Int64 {\n    return a + b;\n}\n";
        let p2 = Lexer::new(src2).tokenize().ok()
            .and_then(|t| Parser::new(t).parse_program().ok())
            .unwrap();
        let span2 = resolve_definition(Some(&p2), (1, 16), "a").unwrap();
        assert_eq!((span2.start.line, span2.start.col), (1, 9));
    }

    #[test]
    fn definition_resolves_function_name_and_ignores_foreign_let() {
        let src = "@fn helper() -> Void {\n    let tmp: Int64 = 1;\n}\n@fn main() -> Void {\n    helper();\n}\n";
        let program = Lexer::new(src).tokenize().ok()
            .and_then(|t| Parser::new(t).parse_program().ok())
            .unwrap();
        // Global function reference.
        let span = resolve_definition(Some(&program), (3, 6), "helper").unwrap();
        assert_eq!((span.start.line, span.start.col), (1, 5));
        // `tmp` from helper() must NOT resolve from a position in main().
        assert!(resolve_definition(Some(&program), (3, 12), "tmp").is_none());
        // Unknown word resolves to nothing, not a false positive.
        assert!(resolve_definition(Some(&program), (3, 12), "zzz").is_none());
    }

    #[test]
    fn definition_resolves_struct_name() {
        let src = "@struct Point { x: Float64, y: Float64 }\n@fn main() -> Void {\n    let p: Point = point(1, 2);\n}\n";
        let program = Lexer::new(src).tokenize().ok()
            .and_then(|t| Parser::new(t).parse_program().ok())
            .unwrap();
        let span = resolve_definition(Some(&program), (2, 12), "Point").unwrap();
        assert_eq!((span.start.line, span.start.col), (1, 9));
    }
}