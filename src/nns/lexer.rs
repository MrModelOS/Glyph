#![allow(dead_code)]
// NNS port: public API preserved for parity with C++ nsc; not all items are used in current pipeline — intentional, not tech debt
//! Lexer — port of `ns/lexer/lexer.cpp`.

use super::token::{Token, TokenType};
use super::NsResult;

fn is_alpha(c: u8) -> bool {
    c.is_ascii_alphabetic()
}

fn is_digit(c: u8) -> bool {
    c.is_ascii_digit()
}

fn is_alnum(c: u8) -> bool {
    c.is_ascii_alphanumeric()
}

fn is_space(c: u8) -> bool {
    c.is_ascii_whitespace()
}

pub struct Lexer {
    source: Vec<u8>,
    pos: usize,
    line: u32,
    column: u32,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Lexer {
            source: source.as_bytes().to_vec(),
            pos: 0,
            line: 1,
            column: 1,
        }
    }

    pub fn tokenize(&mut self) -> NsResult<Vec<Token>> {
        let mut tokens: Vec<Token> = Vec::new();
        loop {
            self.skip_whitespace();
            if self.pos >= self.source.len() {
                break;
            }
            tokens.push(self.next_token()?);
        }
        tokens.push(Token::eof(self.line, self.column));
        Ok(tokens)
    }

    pub fn next_token(&mut self) -> NsResult<Token> {
        self.skip_whitespace();
        if self.pos >= self.source.len() {
            return Ok(self.make_token(TokenType::EofToken, ""));
        }

        let c = self.current_char();
        let peek = self.peek_char();

        if c == b'/' && peek == b'/' {
            self.skip_line_comment();
            return self.next_token();
        }
        if c == b'/' && peek == b'*' {
            self.skip_comment();
            return self.next_token();
        }

        if is_alpha(c) || c == b'_' {
            return Ok(self.read_identifier());
        }
        if is_digit(c) || (c == b'.' && is_digit(peek)) {
            return Ok(self.read_number());
        }
        if c == b'"' {
            return Ok(self.read_string());
        }

        match c {
            b'+' => {
                self.advance();
                Ok(self.make_token(TokenType::OpPlus, ""))
            }
            b'-' => {
                self.advance();
                if self.pos < self.source.len() && self.current_char() == b'>' {
                    self.advance();
                    Ok(self.make_token(TokenType::OpPipeline, ""))
                } else {
                    Ok(self.make_token(TokenType::OpMinus, ""))
                }
            }
            b'*' => {
                self.advance();
                Ok(self.make_token(TokenType::OpStar, ""))
            }
            b'/' => {
                self.advance();
                Ok(self.make_token(TokenType::OpSlash, ""))
            }
            b'%' => {
                self.advance();
                Ok(self.make_token(TokenType::OpPercent, ""))
            }
            b'=' => {
                self.advance();
                if self.pos < self.source.len() && self.current_char() == b'=' {
                    self.advance();
                    Ok(self.make_token(TokenType::OpEq, ""))
                } else if self.pos < self.source.len() && self.current_char() == b'>' {
                    self.advance();
                    Ok(self.make_token(TokenType::OpArrowFunc, ""))
                } else {
                    Ok(self.make_token(TokenType::OpAssign, ""))
                }
            }
            b'!' => {
                self.advance();
                if self.pos < self.source.len() && self.current_char() == b'=' {
                    self.advance();
                    Ok(self.make_token(TokenType::OpNeq, ""))
                } else {
                    Ok(self.make_token(TokenType::OpNot, ""))
                }
            }
            b'<' => {
                self.advance();
                if self.pos < self.source.len() && self.current_char() == b'=' {
                    self.advance();
                    Ok(self.make_token(TokenType::OpLte, ""))
                } else {
                    Ok(self.make_token(TokenType::OpLt, ""))
                }
            }
            b'>' => {
                self.advance();
                if self.pos < self.source.len() && self.current_char() == b'=' {
                    self.advance();
                    Ok(self.make_token(TokenType::OpGte, ""))
                } else {
                    Ok(self.make_token(TokenType::OpGt, ""))
                }
            }
            b'&' => {
                self.advance();
                if self.pos < self.source.len() && self.current_char() == b'&' {
                    self.advance();
                    Ok(self.make_token(TokenType::OpAnd, ""))
                } else {
                    Err(super::NsError(format!(
                        "Unexpected character '&' at line {}",
                        self.line
                    )))
                }
            }
            b'|' => {
                self.advance();
                if self.pos < self.source.len() && self.current_char() == b'|' {
                    self.advance();
                    Ok(self.make_token(TokenType::OpOr, ""))
                } else {
                    Err(super::NsError(format!(
                        "Unexpected character '|' at line {}",
                        self.line
                    )))
                }
            }
            b'@' => {
                self.advance();
                Ok(self.make_token(TokenType::OpMatmul, ""))
            }
            b'.' => {
                self.advance();
                Ok(self.make_token(TokenType::OpDot, ""))
            }
            b',' => {
                self.advance();
                Ok(self.make_token(TokenType::OpComma, ""))
            }
            b':' => {
                self.advance();
                if self.pos < self.source.len() && self.current_char() == b':' {
                    self.advance();
                    Ok(self.make_token(TokenType::OpDoubleColon, ""))
                } else {
                    Ok(self.make_token(TokenType::OpColon, ""))
                }
            }
            b';' => {
                self.advance();
                Ok(self.make_token(TokenType::OpSemicolon, ""))
            }
            b'(' => {
                self.advance();
                Ok(self.make_token(TokenType::OpLparen, ""))
            }
            b')' => {
                self.advance();
                Ok(self.make_token(TokenType::OpRparen, ""))
            }
            b'{' => {
                self.advance();
                Ok(self.make_token(TokenType::OpLbrace, ""))
            }
            b'}' => {
                self.advance();
                Ok(self.make_token(TokenType::OpRbrace, ""))
            }
            b'[' => {
                self.advance();
                Ok(self.make_token(TokenType::OpLbracket, ""))
            }
            b']' => {
                self.advance();
                Ok(self.make_token(TokenType::OpRbracket, ""))
            }
            _ => Err(super::NsError(format!(
                "Unexpected character '{}' at line {}:{}",
                c as char, self.line, self.column
            ))),
        }
    }

    pub fn peek_token(&mut self) -> NsResult<Token> {
        let saved_pos = self.pos;
        let saved_line = self.line;
        let saved_col = self.column;
        let tok = self.next_token()?;
        self.pos = saved_pos;
        self.line = saved_line;
        self.column = saved_col;
        Ok(tok)
    }

    pub fn has_more(&self) -> bool {
        self.pos < self.source.len()
    }

    fn current_char(&self) -> u8 {
        self.source[self.pos]
    }

    fn peek_char(&self) -> u8 {
        if self.pos + 1 >= self.source.len() {
            return 0;
        }
        self.source[self.pos + 1]
    }

    fn advance(&mut self) {
        if self.source[self.pos] == b'\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        self.pos += 1;
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.source.len() && is_space(self.source[self.pos]) {
            self.advance();
        }
    }

    fn skip_comment(&mut self) {
        self.advance(); // skip /
        self.advance(); // skip *
        while self.pos < self.source.len() - 1 {
            if self.source[self.pos] == b'*' && self.source[self.pos + 1] == b'/' {
                self.advance();
                self.advance();
                return;
            }
            self.advance();
        }
    }

    fn skip_line_comment(&mut self) {
        self.advance();
        self.advance(); // skip //
        while self.pos < self.source.len() && self.source[self.pos] != b'\n' {
            self.advance();
        }
    }

    fn read_identifier(&mut self) -> Token {
        let start = self.pos;
        let start_col = self.column;
        while self.pos < self.source.len()
            && (is_alnum(self.source[self.pos]) || self.source[self.pos] == b'_')
        {
            self.advance();
        }
        let word = String::from_utf8_lossy(&self.source[start..self.pos]).into_owned();
        let type_ = Self::keyword_or_identifier(&word);
        Token::new(type_, &word, self.line, start_col)
    }

    fn read_number(&mut self) -> Token {
        let start = self.pos;
        let start_col = self.column;
        let mut is_float = false;

        while self.pos < self.source.len() && is_digit(self.source[self.pos]) {
            self.advance();
        }

        if self.pos < self.source.len()
            && self.source[self.pos] == b'.'
            && self.pos + 1 < self.source.len()
            && is_digit(self.source[self.pos + 1])
        {
            is_float = true;
            self.advance(); // skip .
            while self.pos < self.source.len() && is_digit(self.source[self.pos]) {
                self.advance();
            }
        }

        if self.pos < self.source.len()
            && (self.source[self.pos] == b'e' || self.source[self.pos] == b'E')
        {
            is_float = true;
            self.advance();
            if self.pos < self.source.len()
                && (self.source[self.pos] == b'+' || self.source[self.pos] == b'-')
            {
                self.advance();
            }
            while self.pos < self.source.len() && is_digit(self.source[self.pos]) {
                self.advance();
            }
        }

        let value = String::from_utf8_lossy(&self.source[start..self.pos]).into_owned();
        let t = if is_float {
            TokenType::FloatLiteral
        } else {
            TokenType::IntLiteral
        };
        Token::new(t, &value, self.line, start_col)
    }

    fn read_string(&mut self) -> Token {
        let start_col = self.column;
        self.advance(); // skip opening "
        let mut value = String::new();
        while self.pos < self.source.len() && self.source[self.pos] != b'"' {
            if self.source[self.pos] == b'\\' {
                self.advance();
                if self.pos < self.source.len() {
                    match self.source[self.pos] {
                        b'n' => value.push('\n'),
                        b't' => value.push('\t'),
                        b'\\' => value.push('\\'),
                        b'"' => value.push('"'),
                        c => value.push(c as char),
                    }
                }
            } else {
                value.push(self.source[self.pos] as char);
            }
            self.advance();
        }
        if self.pos < self.source.len() {
            self.advance(); // skip closing "
        }
        Token::new(TokenType::StringLiteral, &value, self.line, start_col)
    }

    fn make_token(&self, type_: TokenType, value: &str) -> Token {
        let col = if value.is_empty() {
            self.column
        } else {
            self.column.saturating_sub(value.len() as u32)
        };
        Token::new(type_, value, self.line, col)
    }

    fn keyword_or_identifier(word: &str) -> TokenType {
        match word {
            "fn" => TokenType::KwFn,
            "var" => TokenType::KwVar,
            "return" => TokenType::KwReturn,
            "if" => TokenType::KwIf,
            "else" => TokenType::KwElse,
            "while" => TokenType::KwWhile,
            "for" => TokenType::KwFor,
            "network" => TokenType::KwNetwork,
            "layer" => TokenType::KwLayer,
            "forward" => TokenType::KwForward,
            "train" => TokenType::KwTrain,
            "grad" => TokenType::KwGrad,
            "mut" => TokenType::KwMut,
            "ref" => TokenType::KwRef,
            "type" => TokenType::KwType,
            "as" => TokenType::KwAs,

            "int" => TokenType::TypeInt,
            "float" => TokenType::TypeFloat,
            "bool" => TokenType::TypeBool,
            "string" => TokenType::TypeString,
            "Tensor" => TokenType::TypeTensor,
            "Dynamic" => TokenType::TypeDynamic,

            "float16" => TokenType::DtypeFloat16,
            "float32" => TokenType::DtypeFloat32,
            "float64" => TokenType::DtypeFloat64,
            "int8" => TokenType::DtypeInt8,
            "int16" => TokenType::DtypeInt16,
            "int32" => TokenType::DtypeInt32,
            "int64" => TokenType::DtypeInt64,
            "fp8" => TokenType::DtypeFp8,
            "fp4" => TokenType::DtypeFp4,

            "ReLU" => TokenType::ActRelu,
            "LeakyReLU" => TokenType::ActLeakyRelu,
            "Sigmoid" => TokenType::ActSigmoid,
            "Tanh" => TokenType::ActTanh,
            "Swish" => TokenType::ActSwish,
            "GELU" => TokenType::ActGelu,
            "SwiGLU" => TokenType::ActSwiglu,
            "SiLU" => TokenType::ActSilu,
            "Identity" => TokenType::ActIdentity,
            "Softmax" => TokenType::ActSoftmax,

            "AdamW" => TokenType::OptAdamw,
            "Muon" => TokenType::OptMuon,
            "SGD" => TokenType::OptSgd,

            "true" | "false" => TokenType::Identifier,

            _ => TokenType::Identifier,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn types(src: &str) -> Vec<TokenType> {
        Lexer::new(src)
            .tokenize()
            .unwrap()
            .into_iter()
            .map(|t| t.type_)
            .collect()
    }

    #[test]
    fn tokenizes_pipeline_network() {
        let src = "network mlp { input: Tensor[Batch, 4]; forward(x) { return x -> fc1 -> fc2; } }";
        let toks = types(src);
        assert!(toks.contains(&TokenType::KwNetwork));
        assert!(toks.contains(&TokenType::KwForward));
        assert!(toks.contains(&TokenType::OpPipeline));
        assert!(toks.contains(&TokenType::TypeTensor));
    }

    #[test]
    fn numbers_float_and_int() {
        assert!(types("x = 10;").contains(&TokenType::IntLiteral));
        assert!(types("x = 3.14;").contains(&TokenType::FloatLiteral));
        assert!(types("x = 1e-3;").contains(&TokenType::FloatLiteral));
    }

    #[test]
    fn optimizer_and_activation_keywords() {
        let src = "AdamW Muon SGD ReLU GELU SwiGLU Identity Softmax";
        let toks = types(src);
        assert!(toks.contains(&TokenType::OptAdamw));
        assert!(toks.contains(&TokenType::OptMuon));
        assert!(toks.contains(&TokenType::OptSgd));
        assert!(toks.contains(&TokenType::ActRelu));
        assert!(toks.contains(&TokenType::ActGelu));
        assert!(toks.contains(&TokenType::ActSwiglu));
        assert!(toks.contains(&TokenType::ActIdentity));
        assert!(toks.contains(&TokenType::ActSoftmax));
    }

    #[test]
    fn strings_with_escapes() {
        let toks = Lexer::new(r#"s = "a\nb";"#).tokenize().unwrap();
        let s = toks
            .iter()
            .find(|t| t.type_ == TokenType::StringLiteral)
            .unwrap();
        assert_eq!(s.value, "a\nb");
    }
}
