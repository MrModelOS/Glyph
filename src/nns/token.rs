#![allow(dead_code)]
//! Token types and the Token struct — port of `ns/lexer/token.hpp` + `token.cpp`.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    // Literals
    IntLiteral,
    FloatLiteral,
    StringLiteral,
    Identifier,

    // Types
    TypeInt,
    TypeFloat,
    TypeBool,
    TypeString,
    TypeTensor,
    TypeDynamic,

    // Keywords
    KwFn,
    KwVar,
    KwReturn,
    KwIf,
    KwElse,
    KwWhile,
    KwFor,
    KwNetwork,
    KwLayer,
    KwForward,
    KwInput,
    KwOutput,
    KwGrad,
    KwMut,
    KwRef,
    KwType,
    KwAs,
    KwTrain,

    // Dtypes
    DtypeFloat16,
    DtypeFloat32,
    DtypeFloat64,
    DtypeInt8,
    DtypeInt16,
    DtypeInt32,
    DtypeInt64,
    DtypeFp8,
    DtypeFp4,
    DtypeBool,

    // Operators
    OpPlus,
    OpMinus,
    OpStar,
    OpSlash,
    OpPercent,
    OpAssign,
    OpEq,
    OpNeq,
    OpLt,
    OpGt,
    OpLte,
    OpGte,
    OpAnd,
    OpOr,
    OpNot,
    OpMatmul,     // @
    OpPipeline,   // ->
    OpArrowFunc,  // =>
    OpDot,
    OpComma,
    OpColon,
    OpSemicolon,
    OpLparen,
    OpRparen,
    OpLbrace,
    OpRbrace,
    OpLbracket,
    OpRbracket,
    OpDoubleColon, // ::

    // Activation functions
    ActRelu,
    ActLeakyRelu,
    ActSigmoid,
    ActTanh,
    ActSwish,
    ActGelu,
    ActSwiglu,
    ActSilu,
    ActIdentity,
    ActSoftmax,

    // Optimizer keywords
    OptAdamw,
    OptMuon,
    OptSgd,

    // Special
    EofToken,
    Error,
}

impl TokenType {
    /// Display name, byte-for-byte as the C++ `token_type_name()`.
    pub fn name(&self) -> &'static str {
        match self {
            TokenType::IntLiteral => "INT_LITERAL",
            TokenType::FloatLiteral => "FLOAT_LITERAL",
            TokenType::StringLiteral => "STRING_LITERAL",
            TokenType::Identifier => "IDENTIFIER",
            TokenType::TypeInt => "int",
            TokenType::TypeFloat => "float",
            TokenType::TypeBool => "bool",
            TokenType::TypeString => "string",
            TokenType::TypeTensor => "Tensor",
            TokenType::TypeDynamic => "Dynamic",
            TokenType::KwFn => "fn",
            TokenType::KwVar => "var",
            TokenType::KwReturn => "return",
            TokenType::KwIf => "if",
            TokenType::KwElse => "else",
            TokenType::KwWhile => "while",
            TokenType::KwFor => "for",
            TokenType::KwNetwork => "network",
            TokenType::KwLayer => "layer",
            TokenType::KwForward => "forward",
            TokenType::KwInput => "input",
            TokenType::KwOutput => "output",
            TokenType::KwGrad => "grad",
            TokenType::KwMut => "mut",
            TokenType::KwRef => "ref",
            TokenType::KwType => "type",
            TokenType::KwAs => "as",
            TokenType::KwTrain => "train",
            TokenType::DtypeFloat16 => "float16",
            TokenType::DtypeFloat32 => "float32",
            TokenType::DtypeFloat64 => "float64",
            TokenType::DtypeInt8 => "int8",
            TokenType::DtypeInt16 => "int16",
            TokenType::DtypeInt32 => "int32",
            TokenType::DtypeInt64 => "int64",
            TokenType::DtypeFp8 => "fp8",
            TokenType::DtypeFp4 => "fp4",
            TokenType::DtypeBool => "bool",
            TokenType::OpPlus => "+",
            TokenType::OpMinus => "-",
            TokenType::OpStar => "*",
            TokenType::OpSlash => "/",
            TokenType::OpPercent => "%",
            TokenType::OpAssign => "=",
            TokenType::OpEq => "==",
            TokenType::OpNeq => "!=",
            TokenType::OpLt => "<",
            TokenType::OpGt => ">",
            TokenType::OpLte => "<=",
            TokenType::OpGte => ">=",
            TokenType::OpAnd => "&&",
            TokenType::OpOr => "||",
            TokenType::OpNot => "!",
            TokenType::OpMatmul => "@",
            TokenType::OpPipeline => "->",
            TokenType::OpArrowFunc => "=>",
            TokenType::OpDot => ".",
            TokenType::OpComma => ",",
            TokenType::OpColon => ":",
            TokenType::OpSemicolon => ";",
            TokenType::OpLparen => "(",
            TokenType::OpRparen => ")",
            TokenType::OpLbrace => "{",
            TokenType::OpRbrace => "}",
            TokenType::OpLbracket => "[",
            TokenType::OpRbracket => "]",
            TokenType::OpDoubleColon => "::",
            TokenType::ActRelu => "ReLU",
            TokenType::ActLeakyRelu => "LeakyReLU",
            TokenType::ActSigmoid => "Sigmoid",
            TokenType::ActTanh => "Tanh",
            TokenType::ActSwish => "Swish",
            TokenType::ActGelu => "GELU",
            TokenType::ActSwiglu => "SwiGLU",
            TokenType::ActSilu => "SiLU",
            TokenType::ActIdentity => "Identity",
            TokenType::ActSoftmax => "Softmax",
            TokenType::OptAdamw => "AdamW",
            TokenType::OptMuon => "Muon",
            TokenType::OptSgd => "SGD",
            TokenType::EofToken => "EOF",
            TokenType::Error => "ERROR",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    pub type_: TokenType,
    pub value: String,
    pub line: u32,
    pub column: u32,
}

impl Token {
    pub fn new(t: TokenType, v: &str, l: u32, c: u32) -> Self {
        Token {
            type_: t,
            value: v.to_string(),
            line: l,
            column: c,
        }
    }

    pub fn eof(line: u32, column: u32) -> Self {
        Token::new(TokenType::EofToken, "", line, column)
    }

    pub fn is(&self, t: TokenType) -> bool {
        self.type_ == t
    }

    pub fn is_one_of(&self, types: &[TokenType]) -> bool {
        types.contains(&self.type_)
    }
}