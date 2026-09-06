use std::collections::HashMap;
use std::io::{self, BufRead, Read, Write};

use crate::ast::*;

#[derive(Debug)]
pub struct LspServer {
    documents: HashMap<String, Document>,
}

#[derive(Debug)]
struct Document {
    content: String,
    version: i32,
    ast: Option<Program>,
    diagnostics: Vec<Diagnostic>,
}

#[derive(Debug)]
struct Diagnostic {
    line: usize,
    column: usize,
    message: String,
    severity: DiagnosticSeverity,
}

#[derive(Debug)]
enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
}

impl LspServer {
    pub fn new() -> Self {
        LspServer {
            documents: HashMap::new(),
        }
    }

    pub fn run(&mut self) {
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut reader = stdin.lock();
        let mut writer = stdout.lock();

        // Simple message loop
        loop {
            let mut header = String::new();
            match reader.read_line(&mut header) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    let header = header.trim();
                    if header.is_empty() {
                        continue;
                    }

                    // Parse Content-Length header
                    if let Some(len_str) = header.strip_prefix("Content-Length: ") {
                        let len: usize = len_str.parse().unwrap_or(0);
                        if len > 0 {
                            // Read the empty line
                            let mut empty_line = String::new();
                            reader.read_line(&mut empty_line).unwrap();

                            // Read the JSON body
                            let mut body = vec![0u8; len];
                            reader.read_exact(&mut body).unwrap();
                            let json_str = String::from_utf8(body).unwrap();

                            // Handle the message
                            let response = self.handle_message(&json_str);

                            // Send response
                            if let Some(resp) = response {
                                let resp_bytes = resp.as_bytes();
                                write!(writer, "Content-Length: {}\r\n\r\n{}", resp_bytes.len(), resp).unwrap();
                                writer.flush().unwrap();
                            }
                        }
                    }
                }
                Err(_) => break,
            }
        }
    }

    fn handle_message(&mut self, json_str: &str) -> Option<String> {
        // Simple JSON parsing for demonstration
        // In a real implementation, use a proper JSON parser

        if json_str.contains("\"initialize\"") {
            return Some(self.handle_initialize());
        }

        if json_str.contains("\"textDocument/didOpen\"") {
            return self.handle_did_open(json_str);
        }

        if json_str.contains("\"textDocument/didChange\"") {
            return self.handle_did_change(json_str);
        }

        if json_str.contains("\"textDocument/completion\"") {
            return self.handle_completion(json_str);
        }

        if json_str.contains("\"textDocument/hover\"") {
            return self.handle_hover(json_str);
        }

        if json_str.contains("\"textDocument/diagnostic\"") {
            return self.handle_diagnostics(json_str);
        }

        None
    }

    fn handle_initialize(&self) -> String {
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "capabilities": {
                    "textDocumentSync": 1,
                    "completionProvider": {
                        "triggerCharacters": ["@", "#", "."]
                    },
                    "hoverProvider": true,
                    "diagnosticProvider": {
                        "interFileDependencies": false,
                        "workspaceDiagnostics": false
                    }
                }
            }
        })
        .to_string()
    }

    fn handle_did_open(&mut self, _json_str: &str) -> Option<String> {
        // Extract document URI and content from JSON
        // For demonstration, use placeholder values
        let uri = "file:///example.glyph";
        let content = "";

        self.documents.insert(
            uri.to_string(),
            Document {
                content: content.to_string(),
                version: 1,
                ast: None,
                diagnostics: Vec::new(),
            },
        );

        Some(serde_json::json!({
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": uri,
                "diagnostics": []
            }
        })
        .to_string())
    }

    fn handle_did_change(&mut self, _json_str: &str) -> Option<String> {
        // Update document content
        // For demonstration, return empty diagnostics
        Some(serde_json::json!({
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": "file:///example.glyph",
                "diagnostics": []
            }
        })
        .to_string())
    }

    fn handle_completion(&self, _json_str: &str) -> Option<String> {
        let completions = vec![
            serde_json::json!({
                "label": "@module",
                "kind": 14,
                "detail": "Module declaration"
            }),
            serde_json::json!({
                "label": "@fn",
                "kind": 14,
                "detail": "Function declaration"
            }),
            serde_json::json!({
                "label": "@struct",
                "kind": 14,
                "detail": "Struct declaration"
            }),
            serde_json::json!({
                "label": "@enum",
                "kind": 14,
                "detail": "Enum declaration"
            }),
            serde_json::json!({
                "label": "@trait",
                "kind": 14,
                "detail": "Trait declaration"
            }),
            serde_json::json!({
                "label": "#guard",
                "kind": 14,
                "detail": "Guard contract"
            }),
        ];

        Some(serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": completions
        })
        .to_string())
    }

    fn handle_hover(&self, _json_str: &str) -> Option<String> {
        Some(serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "contents": {
                    "kind": "markdown",
                    "value": "**Glyph Language**\n\nA systems programming language for LLM code generation."
                }
            }
        })
        .to_string())
    }

    fn handle_diagnostics(&self, _json_str: &str) -> Option<String> {
        Some(serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "kind": "full",
                "items": []
            }
        })
        .to_string())
    }
}
