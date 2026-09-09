use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Module,
    Fn,
    Struct,
    Enum,
    Trait,
    Use,
    Let,
    Mut,
    If,
    Else,
    Match,
    Return,
    For,
    In,
    Loop,
    While,
    Break,
    Continue,
    Spawn,
    Await,
    Async,
    True,
    False,
    As,
    Impl,

    // Contract macros
    Guard,
    Inject,
    Inline,
    NoStd,

    // Select
    Select,
    Timeout,
    Default,

    // Declarations
    Const,
    Pub,
    Test,

    // Types
    TypeString,
    TypeUInt64,
    TypeInt64,
    TypeFloat64,
    TypeBool,
    TypeVoid,
    TypeBytes,
    TypeList,
    TypeMap,
    TypeResult,
    TypeOption,
    TypeAsync,
    TypeChannel,

    // Literals
    Integer(i64),
    Float(f64),
    StringLiteral(String),
    Identifier(String),

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Concat,       // ++
    Eq,
    Neq,
    Lt,
    Gt,
    Le,
    Ge,
    Assign,
    Dot,
    Comma,
    Semicolon,
    Colon,
    DoubleColon,
    Arrow,        // ->
    ArrowLeft,    // <-
    Pipe,         // |
    FatArrow,     // =>
    AmpAmp,       // &&
    PipePipe,     // ||
    Bang,         // !
    Ref,          // &
    DotDot,       // ..
    DotDotEq,     // ..=
    Underscore,   // _

    // Delimiters
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,

    // Special
    Eof,
    Pound,        // # (map literal prefix)
}

#[derive(Debug, Clone)]
pub struct SpannedToken {
    pub token: Token,
    pub line: usize,
    pub column: usize,
    /// One past the last character of the token on `line`.
    pub end_line: usize,
    pub end_column: usize,
}

#[derive(Error, Debug)]
pub enum LexerError {
    #[error("Unexpected character '{0}' at line {1}:{2}")]
    UnexpectedChar(char, usize, usize),

    #[error("Unterminated string at line {0}:{1}")]
    UnterminatedString(usize, usize),

    #[error("Invalid number at line {0}:{1}")]
    InvalidNumber(usize, usize),
}

pub struct Lexer {
    input: Vec<char>,
    pos: usize,
    line: usize,
    column: usize,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        Lexer {
            input: input.chars().collect(),
            pos: 0,
            line: 1,
            column: 1,
        }
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.input.get(self.pos).copied()?;
        self.pos += 1;
        if ch == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(ch)
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else if ch == '/' && self.pos + 1 < self.input.len() {
                let next = self.input[self.pos + 1];
                if next == '/' {
                    // Line comment
                    while let Some(c) = self.advance() {
                        if c == '\n' {
                            break;
                        }
                    }
                } else if next == '*' {
                    // Block comment /* ... */
                    self.advance(); // skip /
                    self.advance(); // skip *
                    while let Some(c) = self.advance() {
                        if c == '*' && self.peek() == Some('/') {
                            self.advance(); // skip */
                            break;
                        }
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    fn read_string(&mut self) -> Result<String, LexerError> {
        let start_line = self.line;
        let start_col = self.column;
        let mut s = String::new();

        while let Some(ch) = self.advance() {
            if ch == '"' {
                return Ok(s);
            }
            if ch == '\\' {
                match self.advance() {
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some('\\') => s.push('\\'),
                    Some('"') => s.push('"'),
                    Some(c) => {
                        s.push('\\');
                        s.push(c);
                    }
                    None => return Err(LexerError::UnterminatedString(start_line, start_col)),
                }
            } else {
                s.push(ch);
            }
        }
        Err(LexerError::UnterminatedString(start_line, start_col))
    }

    fn read_identifier(&mut self) -> String {
        let mut ident = String::new();
        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                ident.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        ident
    }

    fn read_number(&mut self) -> Result<(String, bool), LexerError> {
        let start_line = self.line;
        let start_col = self.column;
        let mut num = String::new();
        let mut is_float = false;

        // Check for hex, octal, binary prefix
        if self.peek() == Some('0') {
            if self.pos + 1 < self.input.len() {
                let next = self.input[self.pos + 1];
                if next == 'x' || next == 'X' {
                    // Hex literal
                    num.push('0');
                    num.push('x');
                    self.advance();
                    self.advance();
                    while let Some(ch) = self.peek() {
                        if ch.is_ascii_hexdigit() || ch == '_' {
                            num.push(ch);
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    return Ok((num, false));
                } else if next == 'o' || next == 'O' {
                    // Octal literal
                    num.push('0');
                    num.push('o');
                    self.advance();
                    self.advance();
                    while let Some(ch) = self.peek() {
                        if matches!(ch, '0'..='7' | '_') {
                            num.push(ch);
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    return Ok((num, false));
                } else if next == 'b' || next == 'B' {
                    // Binary literal
                    num.push('0');
                    num.push('b');
                    self.advance();
                    self.advance();
                    while let Some(ch) = self.peek() {
                        if matches!(ch, '0' | '1' | '_') {
                            num.push(ch);
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    return Ok((num, false));
                }
            }
        }

        // Decimal or float
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() || ch == '_' {
                num.push(ch);
                self.advance();
            } else if ch == '.' && !is_float {
                // Check if next char is digit (float) or dot (range operator)
                if self.pos + 1 < self.input.len() && self.input[self.pos + 1].is_ascii_digit() {
                    is_float = true;
                    num.push(ch);
                    self.advance();
                } else {
                    break;
                }
            } else if (ch == 'e' || ch == 'E') && !is_float {
                // Scientific notation
                is_float = true;
                num.push(ch);
                self.advance();
                if self.peek() == Some('+') || self.peek() == Some('-') {
                    num.push(self.peek().unwrap());
                    self.advance();
                }
            } else {
                break;
            }
        }

        if num.is_empty() {
            return Err(LexerError::InvalidNumber(start_line, start_col));
        }

        Ok((num, is_float))
    }

    pub fn tokenize(&mut self) -> Result<Vec<SpannedToken>, LexerError> {
        let mut tokens = Vec::new();

        loop {
            self.skip_whitespace();

            let line = self.line;
            let column = self.column;

            let token = match self.peek() {
                None => {
                    tokens.push(SpannedToken {
                        token: Token::Eof,
                        line,
                        column,
                        end_line: line,
                        end_column: column,
                    });
                    break;
                }
                Some(ch) => {
                    self.advance();
                    match ch {
                        // Single character tokens
                        '{' => Token::LBrace,
                        '}' => Token::RBrace,
                        '(' => Token::LParen,
                        ')' => Token::RParen,
                        '[' => Token::LBracket,
                        ']' => Token::RBracket,
                        ';' => Token::Semicolon,
                        ',' => Token::Comma,
                        '.' => {
                            if self.peek() == Some('.') {
                                self.advance();
                                if self.peek() == Some('=') {
                                    self.advance();
                                    Token::DotDotEq
                                } else {
                                    Token::DotDot
                                }
                            } else {
                                Token::Dot
                            }
                        }
                        '!' => {
                            if self.peek() == Some('=') {
                                self.advance();
                                Token::Neq
                            } else {
                                Token::Bang
                            }
                        }
                        '+' => {
                            if self.peek() == Some('+') {
                                self.advance();
                                Token::Concat
                            } else {
                                Token::Plus
                            }
                        }
                        '-' => {
                            if self.peek() == Some('>') {
                                self.advance();
                                Token::Arrow
                            } else {
                                Token::Minus
                            }
                        }
                        '*' => Token::Star,
                        '/' => Token::Slash,
                        '%' => Token::Percent,
                        '=' => {
                            if self.peek() == Some('=') {
                                self.advance();
                                Token::Eq
                            } else if self.peek() == Some('>') {
                                self.advance();
                                Token::FatArrow
                            } else {
                                Token::Assign
                            }
                        }
                        '<' => {
                            if self.peek() == Some('=') {
                                self.advance();
                                Token::Le
                            } else if self.peek() == Some('-') {
                                self.advance();
                                Token::ArrowLeft
                            } else {
                                Token::Lt
                            }
                        }
                        '>' => {
                            if self.peek() == Some('=') {
                                self.advance();
                                Token::Ge
                            } else {
                                Token::Gt
                            }
                        }
                        ':' => {
                            if self.peek() == Some(':') {
                                self.advance();
                                Token::DoubleColon
                            } else {
                                Token::Colon
                            }
                        }
                        '|' => {
                            if self.peek() == Some('|') {
                                self.advance();
                                Token::PipePipe
                            } else {
                                Token::Pipe
                            }
                        }
                        '&' => {
                            if self.peek() == Some('&') {
                                self.advance();
                                Token::AmpAmp
                            } else {
                                Token::Ref
                            }
                        }

                        // @ prefixes
                        '@' => {
                            let ident = self.read_identifier();
                            match ident.as_str() {
                                "module" => Token::Module,
                                "fn" => Token::Fn,
                                "struct" => Token::Struct,
                                "enum" => Token::Enum,
                                "trait" => Token::Trait,
                                "use" => Token::Use,
                                "impl" => Token::Impl,
                                "const" => Token::Const,
                                "pub" => Token::Pub,
                                "test" => Token::Test,
                                _ => return Err(LexerError::UnexpectedChar('@', line, column)),
                            }
                        }

                        // # prefixes
                        '#' => {
                            let ident = self.read_identifier();
                            if ident.is_empty() {
                                Token::Pound
                            } else {
                                match ident.as_str() {
                                    "guard" => Token::Guard,
                                    "inject" => Token::Inject,
                                    "inline" => Token::Inline,
                                    "no_std" => Token::NoStd,
                                    _ => return Err(LexerError::UnexpectedChar('#', line, column)),
                                }
                            }
                        }

                        // Strings
                        '"' => {
                            let s = self.read_string()?;
                            Token::StringLiteral(s)
                        }

                        // Numbers
                        '0'..='9' => {
                            self.pos -= 1;
                            self.column -= 1;
                            let (num_str, is_float) = self.read_number()?;
                            if is_float {
                                Token::Float(num_str.parse().map_err(|_| LexerError::InvalidNumber(line, column))?)
                            } else {
                                Token::Integer(num_str.parse().map_err(|_| LexerError::InvalidNumber(line, column))?)
                            }
                        }

                        // Identifiers and keywords
                        'a'..='z' | 'A'..='Z' | '_' => {
                            self.pos -= 1;
                            self.column -= 1;
                            let ident = self.read_identifier();
                            // Check for underscore wildcard
                            if ident == "_" {
                                Token::Underscore
                            } else {
                            match ident.as_str() {
                                "let" => Token::Let,
                                "mut" => Token::Mut,
                                "if" => Token::If,
                                "else" => Token::Else,
                                "match" => Token::Match,
                                "return" => Token::Return,
                                "for" => Token::For,
                                "in" => Token::In,
                                "loop" => Token::Loop,
                                "while" => Token::While,
                                "break" => Token::Break,
                                "continue" => Token::Continue,
                                "spawn" => Token::Spawn,
                                "await" => Token::Await,
                                "select" => Token::Select,
                                "timeout" => Token::Timeout,
                                "default" => Token::Default,
                                "async" => Token::Async,
                                "true" => Token::True,
                                "false" => Token::False,
                                "as" => Token::As,
                                "String" => Token::TypeString,
                                "UInt64" => Token::TypeUInt64,
                                "Int64" => Token::TypeInt64,
                                "Float64" => Token::TypeFloat64,
                                "Bool" => Token::TypeBool,
                                "Void" => Token::TypeVoid,
                                "Bytes" => Token::TypeBytes,
                                "List" => Token::TypeList,
                                "Map" => Token::TypeMap,
                                "Result" => Token::TypeResult,
                                "Option" => Token::TypeOption,
                                "Async" => Token::TypeAsync,
                                "Channel" => Token::TypeChannel,
                                _ => Token::Identifier(ident),
                            }
                            }
                        }

                        _ => return Err(LexerError::UnexpectedChar(ch, line, column)),
                    }
                }
            };

            tokens.push(SpannedToken {
                token,
                line,
                column,
                end_line: self.line,
                end_column: self.column,
            });
        }

        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let mut lexer = Lexer::new("@fn main() -> Void {}");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].token, Token::Fn);
        assert_eq!(tokens[1].token, Token::Identifier("main".to_string()));
        assert_eq!(tokens[2].token, Token::LParen);
        assert_eq!(tokens[3].token, Token::RParen);
        assert_eq!(tokens[4].token, Token::Arrow);
        assert_eq!(tokens[5].token, Token::TypeVoid);
        assert_eq!(tokens[6].token, Token::LBrace);
        assert_eq!(tokens[7].token, Token::RBrace);
    }

    #[test]
    fn test_string_literal() {
        let mut lexer = Lexer::new(r#""hello world""#);
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].token, Token::StringLiteral("hello world".to_string()));
    }

    #[test]
    fn test_number_literals() {
        let mut lexer = Lexer::new("42 3.14");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].token, Token::Integer(42));
        assert_eq!(tokens[1].token, Token::Float(3.14));
    }

    #[test]
    fn test_contract_macros() {
        let mut lexer = Lexer::new("#guard(x > 0) #inline");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].token, Token::Guard);
        assert_eq!(tokens[6].token, Token::Inline);
    }

    #[test]
    fn test_test_attribute() {
        let mut lexer = Lexer::new("@test @fn test_add() -> Void {}");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].token, Token::Test);
        assert_eq!(tokens[1].token, Token::Fn);
        assert_eq!(tokens[2].token, Token::Identifier("test_add".to_string()));
    }

    #[test]
    fn test_operators() {
        let mut lexer = Lexer::new("a + b ++ c == d != e");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].token, Token::Identifier("a".to_string()));
        assert_eq!(tokens[1].token, Token::Plus);
        assert_eq!(tokens[2].token, Token::Identifier("b".to_string()));
        assert_eq!(tokens[3].token, Token::Concat);
        assert_eq!(tokens[4].token, Token::Identifier("c".to_string()));
        assert_eq!(tokens[5].token, Token::Eq);
        assert_eq!(tokens[6].token, Token::Identifier("d".to_string()));
        assert_eq!(tokens[7].token, Token::Neq);
        assert_eq!(tokens[8].token, Token::Identifier("e".to_string()));
    }
}
