#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    // Primitive types
    String,
    UInt64,
    Int64,
    Float64,
    Bool,
    Void,
    Bytes,

    // Generic types
    List(Box<Type>),
    Map(Box<Type>, Box<Type>),
    Result(Box<Type>, Box<Type>),
    Option(Box<Type>),
    Async(Box<Type>),
    Channel(Box<Type>),

    // Array type (fixed size)
    Array(Box<Type>, usize),

    // Reference type
    Ref(Box<Type>),

    // User-defined type
    Custom(String),

    // Type parameter of a generic function
    Generic(String),
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::String => write!(f, "String"),
            Type::UInt64 => write!(f, "UInt64"),
            Type::Int64 => write!(f, "Int64"),
            Type::Float64 => write!(f, "Float64"),
            Type::Bool => write!(f, "Bool"),
            Type::Void => write!(f, "Void"),
            Type::Bytes => write!(f, "Bytes"),
            Type::List(inner) => write!(f, "List<{}>", inner),
            Type::Map(k, v) => write!(f, "Map<{}, {}>", k, v),
            Type::Result(ok, err) => write!(f, "Result<{}, {}>", ok, err),
            Type::Option(inner) => write!(f, "Option<{}>", inner),
            Type::Async(inner) => write!(f, "Async<{}>", inner),
            Type::Channel(inner) => write!(f, "Channel<{}>", inner),
            Type::Array(inner, size) => write!(f, "[{}; {}]", inner, size),
            Type::Ref(inner) => write!(f, "&{}", inner),
            Type::Custom(name) => write!(f, "{}", name),
            Type::Generic(name) => write!(f, "{}", name),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // Literals
    IntegerLiteral(i64, Span),
    FloatLiteral(f64, Span),
    StringLiteral(String, Span),
    BoolLiteral(bool, Span),

    // Identifier
    Identifier(String, Span),

    // Binary operations
    BinaryOp {
        span: Span,
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    // Unary operations
    UnaryOp {
        span: Span,
        op: UnaryOp,
        expr: Box<Expr>,
    },

    // Reference expression
    Ref(Box<Expr>, Span),

    // Cast expression (as)
    Cast {
        span: Span,
        expr: Box<Expr>,
        target_type: Type,
    },

    // Function call
    FunctionCall {
        span: Span,
        name: Box<Expr>,
        args: Vec<Expr>,
    },

    // Method call
    MethodCall {
        span: Span,
        object: Box<Expr>,
        method: String,
        args: Vec<Expr>,
    },

    // Field access
    FieldAccess {
        span: Span,
        object: Box<Expr>,
        field: String,
    },

    // Index access
    IndexAccess {
        span: Span,
        object: Box<Expr>,
        index: Box<Expr>,
    },

    // Struct initialization
    StructInit {
        span: Span,
        name: String,
        fields: Vec<(String, Expr)>,
    },

    // Enum variant initialization: Shape::Circle(5.0)
    EnumInit {
        span: Span,
        enum_name: String,
        variant: String,
        args: Vec<Expr>,
    },

    // Match expression
    Match {
        span: Span,
        expr: Box<Expr>,
        arms: Vec<MatchArm>,
    },

    // If expression
    If {
        span: Span,
        condition: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Option<Box<Expr>>,
    },

    // Block expression
    Block(Vec<Stmt>, Span),

    // Range expression (..)
    Range {
        span: Span,
        start: Box<Expr>,
        end: Box<Expr>,
        inclusive: bool,
    },

    // Array literal [1, 2, 3]
    ArrayLiteral(Vec<Expr>, Span),

    // Map literal: #{ "key": value, ... }
    MapLiteral(Vec<(Expr, Expr)>, Span),

    // Channel operations
    ChannelBounded {
        span: Span,
        elem_type: Box<Type>,
        capacity: Box<Expr>,
    },

    // Await expression
    Await(Box<Expr>, Span),
}

impl Expr {
    /// Full source span of this expression.
    pub fn span(&self) -> Span {
        match self {
            Expr::IntegerLiteral(_, s)
            | Expr::FloatLiteral(_, s)
            | Expr::StringLiteral(_, s)
            | Expr::BoolLiteral(_, s)
            | Expr::Identifier(_, s)
            | Expr::Ref(_, s)
            | Expr::Block(_, s)
            | Expr::ArrayLiteral(_, s)
            | Expr::MapLiteral(_, s)
            | Expr::Await(_, s) => *s,
            Expr::BinaryOp { span: s, .. }
            | Expr::UnaryOp { span: s, .. }
            | Expr::Cast { span: s, .. }
            | Expr::FunctionCall { span: s, .. }
            | Expr::MethodCall { span: s, .. }
            | Expr::FieldAccess { span: s, .. }
            | Expr::IndexAccess { span: s, .. }
            | Expr::StructInit { span: s, .. }
            | Expr::EnumInit { span: s, .. }
            | Expr::Match { span: s, .. }
            | Expr::If { span: s, .. }
            | Expr::Range { span: s, .. }
            | Expr::ChannelBounded { span: s, .. } => *s,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Concat,
    Eq,
    Neq,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
}

impl std::fmt::Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            BinOp::Concat => "++",
            BinOp::Eq => "==",
            BinOp::Neq => "!=",
            BinOp::Lt => "<",
            BinOp::Gt => ">",
            BinOp::Le => "<=",
            BinOp::Ge => ">=",
            BinOp::And => "&&",
            BinOp::Or => "||",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    // Literal pattern
    IntegerLiteral(i64),
    StringLiteral(String),
    BoolLiteral(bool),

    // Identifier pattern (binding)
    Identifier(String),

    // Enum variant pattern
    EnumVariant {
        enum_name: Option<String>,
        variant: String,
        data: Option<Vec<Pattern>>,
    },

    // Wildcard pattern
    Wildcard,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineCol {
    pub line: usize,
    pub col: usize,
}

impl std::fmt::Display for LineCol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

/// A full source span: the start and end positions of an AST node.
/// `end` is exclusive (one past the last character of the node).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    pub start: LineCol,
    pub end: LineCol,
}

impl Span {
    pub fn new(start: LineCol, end: LineCol) -> Self {
        Span { start, end }
    }

    /// A zero-width span pointing at a single position.
    #[allow(dead_code)] // used by the LSP for cursor diagnostics.
    pub fn point(loc: LineCol) -> Self {
        Span { start: loc, end: loc }
    }
}

impl std::fmt::Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.start == self.end {
            write!(f, "{}:{}", self.start.line, self.start.col)
        } else if self.start.line == self.end.line {
            write!(f, "{}:{}-{}", self.start.line, self.start.col, self.end.col)
        } else {
            write!(f, "{}:{}-{}:{}", self.start.line, self.start.col, self.end.line, self.end.col)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    // Variable declaration
    Let {
        loc: LineCol,
        name: String,
        name_span: Span,
        ty: Option<Type>,
        value: Expr,
        mutable: bool,
    },

    // Assignment
    Assignment {
        loc: LineCol,
        target: Expr,
        value: Expr,
    },

    // Expression statement
    Expression(LineCol, Expr),

    // Return statement
    Return(LineCol, Option<Expr>),

    // Break statement
    Break(LineCol),

    // Continue statement
    Continue(LineCol),

    // Loop statement
    Loop(LineCol, Vec<Stmt>),

    // While loop
    While {
        loc: LineCol,
        condition: Expr,
        body: Vec<Stmt>,
    },

    // For loop
    For {
        loc: LineCol,
        variable: String,
        var_span: Span,
        iterable: Expr,
        body: Vec<Stmt>,
    },

    // Guard statement
    Guard {
        loc: LineCol,
        condition: Expr,
        else_body: Vec<Stmt>,
    },

    // Spawn statement
    Spawn(LineCol, Expr),

    // Select statement: waits for one of several recv/await events, a
    // timeout, or a default arm that fires immediately when nothing is ready.
    Select {
        loc: LineCol,
        arms: Vec<SelectArm>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum SelectEvent {
    /// Fire when the channel can hand over a value without blocking.
    /// `name: ty` binds the received value inside the arm body.
    Recv {
        target: Expr,
        name: String,
        ty: Type,
    },
    /// Fire when the async handle has completed; binds its result.
    Await {
        target: Expr,
        name: String,
        ty: Type,
    },
    /// Fire when `ms` milliseconds have passed and no event fired yet.
    Timeout(Expr),
    /// Fire immediately when no other arm is ready right now.
    Default,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectArm {
    pub event: SelectEvent,
    pub body: Vec<Stmt>,
}

impl Stmt {
    /// Full location of the statement start.
    pub fn loc(&self) -> LineCol {
        match self {
            Stmt::Let { loc: l, .. }
            | Stmt::Assignment { loc: l, .. }
            | Stmt::While { loc: l, .. }
            | Stmt::For { loc: l, .. }
            | Stmt::Guard { loc: l, .. } => *l,
            Stmt::Expression(l, _)
            | Stmt::Return(l, _)
            | Stmt::Break(l)
            | Stmt::Continue(l)
            | Stmt::Loop(l, _)
            | Stmt::Spawn(l, _) => *l,
            Stmt::Select { loc, .. } => *loc,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionParam {
    pub name: String,
    pub name_span: Span,
    pub ty: Type,
    pub is_move: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TopLevelItem {
    // Module declaration
    Module {
        path: Vec<String>,
    },

    // Use declaration
    Use {
        path: Vec<String>,
    },

    // Function declaration
    Function {
        name: String,
        name_span: Span,
        type_params: Vec<String>,
        params: Vec<FunctionParam>,
        return_type: Option<Type>,
        body: Vec<Stmt>,
        is_async: bool,
        is_test: bool,
        pub_vis: Visibility,
    },

    // Struct declaration
    Struct {
        name: String,
        name_span: Span,
        fields: Vec<(String, Type)>,
        pub_vis: Visibility,
    },

    // Enum declaration
    Enum {
        name: String,
        variants: Vec<EnumVariant>,
        pub_vis: Visibility,
    },

    // Trait declaration
    Trait {
        name: String,
        methods: Vec<TraitMethod>,
        pub_vis: Visibility,
    },

    // Impl block
    Impl {
        type_name: String,
        methods: Vec<ImplMethod>,
        pub_vis: Visibility,
    },

    // Const declaration
    Const {
        name: String,
        ty: Type,
        value: Expr,
        pub_vis: Visibility,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub name: String,
    pub value: Option<i64>,
    pub data: Option<Vec<Type>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraitMethod {
    pub name: String,
    pub params: Vec<FunctionParam>,
    pub return_type: Option<Type>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImplMethod {
    pub name: String,
    pub params: Vec<FunctionParam>,
    pub return_type: Option<Type>,
    pub body: Vec<Stmt>,
    pub is_async: bool,
}

/// Visibility modifier for exported items
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Visibility {
    Private,
    Public,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub items: Vec<TopLevelItem>,
}
