#![allow(dead_code)]
//! AST types — port of `ns/parser/ast.hpp` + `ast.cpp`.

use super::token::{Token, TokenType};

// ---- Type System Nodes ----

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Fp8,
    Fp4,
    Bool,
}

impl Dtype {
    pub fn name(&self) -> &'static str {
        match self {
            Dtype::Float16 => "float16",
            Dtype::Float32 => "float32",
            Dtype::Float64 => "float64",
            Dtype::Int8 => "int8",
            Dtype::Int16 => "int16",
            Dtype::Int32 => "int32",
            Dtype::Int64 => "int64",
            Dtype::Fp8 => "fp8",
            Dtype::Fp4 => "fp4",
            Dtype::Bool => "bool",
        }
    }

    pub fn from_token(tt: TokenType) -> Dtype {
        match tt {
            TokenType::DtypeFloat16 => Dtype::Float16,
            TokenType::DtypeFloat32 => Dtype::Float32,
            TokenType::DtypeFloat64 => Dtype::Float64,
            TokenType::DtypeInt8 => Dtype::Int8,
            TokenType::DtypeInt16 => Dtype::Int16,
            TokenType::DtypeInt32 => Dtype::Int32,
            TokenType::DtypeInt64 => Dtype::Int64,
            TokenType::DtypeFp8 => Dtype::Fp8,
            TokenType::DtypeFp4 => Dtype::Fp4,
            TokenType::DtypeBool => Dtype::Bool,
            _ => Dtype::Float32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimExprKind {
    Const,
    Symbolic,
    Dynamic,
}

#[derive(Debug, Clone)]
pub struct DimExpr {
    pub kind: DimExprKind,
    pub const_value: i64,
    pub symbolic_name: String,
}

impl DimExpr {
    pub fn new(kind: DimExprKind, const_value: i64, symbolic_name: String) -> Self {
        DimExpr {
            kind,
            const_value,
            symbolic_name,
        }
    }

    pub fn dynamic() -> Self {
        DimExpr::new(DimExprKind::Dynamic, 0, String::new())
    }

    pub fn constant(v: i64) -> Self {
        DimExpr::new(DimExprKind::Const, v, String::new())
    }

    pub fn symbolic(name: &str) -> Self {
        DimExpr::new(DimExprKind::Symbolic, 0, name.to_string())
    }

    pub fn is_dynamic(&self) -> bool {
        self.kind == DimExprKind::Dynamic
    }

    pub fn is_const(&self) -> bool {
        self.kind == DimExprKind::Const
    }

    pub fn is_symbolic(&self) -> bool {
        self.kind == DimExprKind::Symbolic
    }

    pub fn to_string(&self) -> String {
        match self.kind {
            DimExprKind::Const => self.const_value.to_string(),
            DimExprKind::Symbolic => self.symbolic_name.clone(),
            DimExprKind::Dynamic => "Dynamic".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorType {
    pub dims: Vec<DimExpr>,
    pub dtype: Dtype,
    pub is_mutable: bool,
}

impl TensorType {
    pub fn new(dims: Vec<DimExpr>, dtype: Dtype) -> Self {
        TensorType {
            dims,
            dtype,
            is_mutable: false,
        }
    }

    pub fn is_dynamic(&self) -> bool {
        self.dims.iter().any(|d| d.is_dynamic())
    }

    pub fn to_string(&self) -> String {
        let mut result = String::from("Tensor[");
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                result.push_str(", ");
            }
            result.push_str(&dim.to_string());
        }
        result.push_str("] ");
        result.push_str(self.dtype.name());
        result
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeKind {
    Tensor,
    Scalar,
    Function,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct TypeNode {
    pub kind: TypeKind,
    pub tensor_type: TensorType,
    pub scalar_type: Dtype,
}

impl TypeNode {
    pub fn new(kind: TypeKind, tensor_type: TensorType, scalar_type: Dtype) -> Self {
        TypeNode {
            kind,
            tensor_type,
            scalar_type,
        }
    }

    pub fn tensor(tt: TensorType) -> Self {
        TypeNode::new(TypeKind::Tensor, tt, Dtype::Float32)
    }

    pub fn scalar(dt: Dtype) -> Self {
        TypeNode::new(TypeKind::Scalar, TensorType::new(Vec::new(), Dtype::Float32), dt)
    }

    pub fn unknown() -> Self {
        TypeNode::new(TypeKind::Unknown, TensorType::new(Vec::new(), Dtype::Float32), Dtype::Float32)
    }

    pub fn is_tensor(&self) -> bool {
        self.kind == TypeKind::Tensor
    }

    pub fn is_scalar(&self) -> bool {
        self.kind == TypeKind::Scalar
    }

    pub fn is_dynamic(&self) -> bool {
        self.kind == TypeKind::Tensor && self.tensor_type.is_dynamic()
    }
}

// ---- Expression Nodes ----

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExprKind {
    LiteralInt,
    LiteralFloat,
    LiteralString,
    LiteralBool,
    Identifier,
    BinaryOp,
    UnaryOp,
    MatmulOp,
    PipelineOp,
    FunctionCall,
    IndexOp,
    TypeCast,
    TensorLiteral,
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub token: Token,

    // Binary/unary
    pub left: Option<Box<Expr>>,
    pub right: Option<Box<Expr>>,
    pub operand: Option<Box<Expr>>,

    // Function call
    pub callee: String,
    pub args: Vec<Box<Expr>>,

    // Index
    pub indices: Vec<Box<Expr>>,

    // Type inference result
    pub inferred_type: Option<Box<TypeNode>>,
}

impl Expr {
    pub fn new(kind: ExprKind, token: Token) -> Self {
        Expr {
            kind,
            token,
            left: None,
            right: None,
            operand: None,
            callee: String::new(),
            args: Vec::new(),
            indices: Vec::new(),
            inferred_type: None,
        }
    }
}

// ---- Statement Nodes ----

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StmtKind {
    VarDecl,
    VarAssign,
    ReturnStmt,
    ExprStmt,
    IfStmt,
    WhileStmt,
    Block,
    FnDecl,
    NetworkDecl,
    LayerDecl,
    ForwardDecl,
    GradBlock,
    TypeDecl,
    ImportStmt,
    TrainDecl,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub type_: Option<Box<TypeNode>>,
    pub is_mut: bool,
    pub is_ref: bool,
}

#[derive(Debug, Clone)]
pub struct LayerParam {
    pub name: String,
    pub value: Option<Box<Expr>>,
}

#[derive(Debug, Clone)]
pub struct Stmt {
    pub kind: StmtKind,
    pub token: Token,

    // Variable declaration
    pub var_name: String,
    pub var_type: Option<Box<TypeNode>>,
    pub init_expr: Option<Box<Expr>>,

    // Function declaration / forward decl
    pub fn_name: String,
    pub params: Vec<Param>,
    pub return_type: Option<Box<TypeNode>>,
    pub body: Option<Box<Stmt>>,

    // Network declaration
    pub network_name: String,
    pub layers: Vec<Box<Stmt>>,
    pub methods: Vec<Box<Stmt>>,

    // Layer declaration
    pub layer_name: String,
    pub layer_type: String,
    pub layer_params: Vec<LayerParam>,

    // If/While
    pub condition: Option<Box<Expr>>,
    pub else_branch: Option<Box<Stmt>>,

    // Block
    pub statements: Vec<Box<Stmt>>,

    // Grad block
    pub grad_body: Option<Box<Stmt>>,

    // Type alias
    pub alias_name: String,
    pub alias_type: Option<Box<TypeNode>>,
    pub alias_expr: Option<Box<Expr>>,

    // Expression statement
    pub expr: Option<Box<Expr>>,
}

impl Stmt {
    pub fn new(kind: StmtKind, token: Token) -> Self {
        Stmt {
            kind,
            token,
            var_name: String::new(),
            var_type: None,
            init_expr: None,
            fn_name: String::new(),
            params: Vec::new(),
            return_type: None,
            body: None,
            network_name: String::new(),
            layers: Vec::new(),
            methods: Vec::new(),
            layer_name: String::new(),
            layer_type: String::new(),
            layer_params: Vec::new(),
            condition: None,
            else_branch: None,
            statements: Vec::new(),
            grad_body: None,
            alias_name: String::new(),
            alias_type: None,
            alias_expr: None,
            expr: None,
        }
    }
}

// ---- Program ----

#[derive(Debug, Clone, Default)]
pub struct Program {
    pub top_level: Vec<Box<Stmt>>,
}

impl Program {
    pub fn new() -> Self {
        Program {
            top_level: Vec::new(),
        }
    }

    pub fn add(&mut self, stmt: Box<Stmt>) {
        self.top_level.push(stmt);
    }
}