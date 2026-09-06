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

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // Literals
    IntegerLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BoolLiteral(bool),

    // Identifier
    Identifier(String),

    // Binary operations
    BinaryOp {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    // Unary operations
    UnaryOp {
        op: UnaryOp,
        expr: Box<Expr>,
    },

    // Reference expression
    Ref(Box<Expr>),

    // Cast expression (as)
    Cast {
        expr: Box<Expr>,
        target_type: Type,
    },

    // Function call
    FunctionCall {
        name: Box<Expr>,
        args: Vec<Expr>,
    },

    // Method call
    MethodCall {
        object: Box<Expr>,
        method: String,
        args: Vec<Expr>,
    },

    // Field access
    FieldAccess {
        object: Box<Expr>,
        field: String,
    },

    // Index access
    IndexAccess {
        object: Box<Expr>,
        index: Box<Expr>,
    },

    // Struct initialization
    StructInit {
        name: String,
        fields: Vec<(String, Expr)>,
    },

    // Enum variant initialization: Shape::Circle(5.0)
    EnumInit {
        enum_name: String,
        variant: String,
        args: Vec<Expr>,
    },

    // Match expression
    Match {
        expr: Box<Expr>,
        arms: Vec<MatchArm>,
    },

    // If expression
    If {
        condition: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Option<Box<Expr>>,
    },

    // Block expression
    Block(Vec<Stmt>),

    // Range expression (..)
    Range {
        start: Box<Expr>,
        end: Box<Expr>,
        inclusive: bool,
    },

    // Array literal [1, 2, 3]
    ArrayLiteral(Vec<Expr>),

    // Channel operations
    ChannelBounded {
        capacity: Box<Expr>,
    },

    // Await expression
    Await(Box<Expr>),

    // String literal (hex escape sequences)
    HexLiteral(i64),
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

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    // Variable declaration
    Let {
        name: String,
        ty: Option<Type>,
        value: Expr,
        mutable: bool,
    },

    // Assignment
    Assignment {
        target: Expr,
        value: Expr,
    },

    // Expression statement
    Expression(Expr),

    // Return statement
    Return(Option<Expr>),

    // Break statement
    Break,

    // Continue statement
    Continue,

    // Loop statement
    Loop(Vec<Stmt>),

    // While loop
    While {
        condition: Expr,
        body: Vec<Stmt>,
    },

    // For loop
    For {
        variable: String,
        iterable: Expr,
        body: Vec<Stmt>,
    },

    // Guard statement
    Guard {
        condition: Expr,
        else_body: Vec<Stmt>,
    },

    // Spawn statement
    Spawn(Expr),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionParam {
    pub name: String,
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
