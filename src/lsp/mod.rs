use std::collections::HashMap;
use std::io::{self, BufRead, Read, Write};

use serde_json::{json, Value};

use crate::ast::{LineCol, Program, Span};
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
        self.documents
            .insert(uri.to_string(), Document { content: content.to_string() });
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
}