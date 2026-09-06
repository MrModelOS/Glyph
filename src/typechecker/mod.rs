use crate::ast::*;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TypeError {
    #[error("Type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    #[error("Undefined variable: {0}")]
    UndefinedVariable(String),

    #[error("Undefined function: {0}")]
    UndefinedFunction(String),

    #[error("Undefined type: {0}")]
    UndefinedType(String),

    #[error("Wrong number of arguments: expected {expected}, found {found}")]
    WrongArgumentCount { expected: usize, found: usize },

    #[error("Cannot apply operator to type: {0}")]
    CannotApplyOperator(String),

    #[error("Cannot call non-function: {0}")]
    CannotCallNonFunction(String),

    #[error("Missing return type")]
    MissingReturnType,

    #[error("Guard condition must be Bool")]
    GuardConditionNotBool,

    #[error("For loop iterable must be List, Map, or Range")]
    InvalidIterableType,

    #[error("Cannot use move on non-owning reference")]
    InvalidMove,

    #[error("Invalid cast from {from} to {to}")]
    InvalidCast { from: String, to: String },

    #[error("While condition must be Bool")]
    WhileConditionNotBool,

    #[error("Match not exhaustive: missing variants")]
    NonExhaustiveMatch,

    #[error("Array type mismatch: expected {expected}, found {found}")]
    ArrayTypeMismatch { expected: String, found: String },

    #[error("Const type mismatch: expected {expected}, found {found}")]
    ConstTypeMismatch { expected: String, found: String },
}

#[derive(Debug, Clone)]
pub struct TypeEnv {
    variables: HashMap<String, Type>,
    functions: HashMap<String, FunctionSignature>,
    types: HashMap<String, TypeDefinition>,
    constants: HashMap<String, Type>,
}

#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub params: Vec<(String, Type, bool)>,
    pub return_type: Option<Type>,
    pub is_async: bool,
}

#[derive(Debug, Clone)]
pub enum TypeDefinition {
    Struct { fields: Vec<(String, Type)> },
    Enum { variants: Vec<EnumVariantInfo> },
    Trait { methods: Vec<(String, Vec<(String, Type)>, Option<Type>)> },
    Impl { type_name: String, methods: Vec<(String, Vec<(String, Type)>, Option<Type>)> },
}

#[derive(Debug, Clone)]
pub struct EnumVariantInfo {
    pub name: String,
    pub data: Option<Vec<Type>>,
}

impl TypeEnv {
    pub fn new() -> Self {
        TypeEnv {
            variables: HashMap::new(),
            functions: HashMap::new(),
            types: HashMap::new(),
            constants: HashMap::new(),
        }
    }

    pub fn define_variable(&mut self, name: String, ty: Type) {
        self.variables.insert(name, ty);
    }

    pub fn get_variable(&self, name: &str) -> Option<&Type> {
        self.variables.get(name)
    }

    pub fn define_function(&mut self, name: String, sig: FunctionSignature) {
        self.functions.insert(name, sig);
    }

    pub fn get_function(&self, name: &str) -> Option<&FunctionSignature> {
        self.functions.get(name)
    }

    pub fn define_type(&mut self, name: String, def: TypeDefinition) {
        self.types.insert(name, def);
    }

    pub fn get_type(&self, name: &str) -> Option<&TypeDefinition> {
        self.types.get(name)
    }

    pub fn define_constant(&mut self, name: String, ty: Type) {
        self.constants.insert(name, ty);
    }

    pub fn get_constant(&self, name: &str) -> Option<&Type> {
        self.constants.get(name)
    }
}

pub struct TypeChecker {
    env: TypeEnv,
}

impl TypeChecker {
    pub fn new() -> Self {
        let mut env = TypeEnv::new();

        // Define built-in functions
        env.define_function(
            "print".to_string(),
            FunctionSignature {
                params: vec![("msg".to_string(), Type::String, false)],
                return_type: Some(Type::Void),
                is_async: false,
            },
        );

        // Define Option and Result enum variants
        env.define_type(
            "Option".to_string(),
            TypeDefinition::Enum {
                variants: vec![
                    EnumVariantInfo { name: "Some".to_string(), data: None },
                    EnumVariantInfo { name: "None".to_string(), data: None },
                ],
            },
        );
        env.define_type(
            "Result".to_string(),
            TypeDefinition::Enum {
                variants: vec![
                    EnumVariantInfo { name: "Ok".to_string(), data: None },
                    EnumVariantInfo { name: "Err".to_string(), data: None },
                ],
            },
        );

        // std.math
        env.define_function("abs".to_string(), FunctionSignature { params: vec![("x".into(), Type::Float64, false)], return_type: Some(Type::Float64), is_async: false });
        env.define_function("min".to_string(), FunctionSignature { params: vec![("a".into(), Type::Float64, false), ("b".into(), Type::Float64, false)], return_type: Some(Type::Float64), is_async: false });
        env.define_function("max".to_string(), FunctionSignature { params: vec![("a".into(), Type::Float64, false), ("b".into(), Type::Float64, false)], return_type: Some(Type::Float64), is_async: false });
        env.define_function("clamp".to_string(), FunctionSignature { params: vec![("value".into(), Type::Float64, false), ("low".into(), Type::Float64, false), ("high".into(), Type::Float64, false)], return_type: Some(Type::Float64), is_async: false });
        env.define_function("sqrt".to_string(), FunctionSignature { params: vec![("x".into(), Type::Float64, false)], return_type: Some(Type::Float64), is_async: false });
        env.define_function("pow".to_string(), FunctionSignature { params: vec![("base".into(), Type::Float64, false), ("exp".into(), Type::Float64, false)], return_type: Some(Type::Float64), is_async: false });
        env.define_function("floor".to_string(), FunctionSignature { params: vec![("x".into(), Type::Float64, false)], return_type: Some(Type::Int64), is_async: false });
        env.define_function("ceil".to_string(), FunctionSignature { params: vec![("x".into(), Type::Float64, false)], return_type: Some(Type::Int64), is_async: false });
        env.define_function("round".to_string(), FunctionSignature { params: vec![("x".into(), Type::Float64, false)], return_type: Some(Type::Int64), is_async: false });
        env.define_function("log".to_string(), FunctionSignature { params: vec![("x".into(), Type::Float64, false)], return_type: Some(Type::Float64), is_async: false });
        env.define_function("log2".to_string(), FunctionSignature { params: vec![("x".into(), Type::Float64, false)], return_type: Some(Type::Float64), is_async: false });
        env.define_function("log10".to_string(), FunctionSignature { params: vec![("x".into(), Type::Float64, false)], return_type: Some(Type::Float64), is_async: false });
        env.define_function("sin".to_string(), FunctionSignature { params: vec![("x".into(), Type::Float64, false)], return_type: Some(Type::Float64), is_async: false });
        env.define_function("cos".to_string(), FunctionSignature { params: vec![("x".into(), Type::Float64, false)], return_type: Some(Type::Float64), is_async: false });
        env.define_function("tan".to_string(), FunctionSignature { params: vec![("x".into(), Type::Float64, false)], return_type: Some(Type::Float64), is_async: false });

        // std.io
        env.define_function("println".to_string(), FunctionSignature { params: vec![("msg".into(), Type::String, false)], return_type: Some(Type::Void), is_async: false });
        env.define_function("eprintln".to_string(), FunctionSignature { params: vec![("msg".into(), Type::String, false)], return_type: Some(Type::Void), is_async: false });
        env.define_function("print_int".to_string(), FunctionSignature { params: vec![("value".into(), Type::Int64, false)], return_type: Some(Type::Void), is_async: false });
        env.define_function("print_float".to_string(), FunctionSignature { params: vec![("value".into(), Type::Float64, false)], return_type: Some(Type::Void), is_async: false });
        env.define_function("print_bool".to_string(), FunctionSignature { params: vec![("value".into(), Type::Bool, false)], return_type: Some(Type::Void), is_async: false });
        env.define_function("read_line".to_string(), FunctionSignature { params: vec![], return_type: Some(Type::String), is_async: false });

        // std.string
        env.define_function("len".to_string(), FunctionSignature { params: vec![("s".into(), Type::String, false)], return_type: Some(Type::Int64), is_async: false });
        env.define_function("substring".to_string(), FunctionSignature { params: vec![("s".into(), Type::String, false), ("start".into(), Type::Int64, false), ("end".into(), Type::Int64, false)], return_type: Some(Type::String), is_async: false });
        env.define_function("contains".to_string(), FunctionSignature { params: vec![("s".into(), Type::String, false), ("sub".into(), Type::String, false)], return_type: Some(Type::Bool), is_async: false });
        env.define_function("replace".to_string(), FunctionSignature { params: vec![("s".into(), Type::String, false), ("from".into(), Type::String, false), ("to".into(), Type::String, false)], return_type: Some(Type::String), is_async: false });
        env.define_function("split".to_string(), FunctionSignature { params: vec![("s".into(), Type::String, false), ("delimiter".into(), Type::String, false)], return_type: Some(Type::String), is_async: false });
        env.define_function("trim".to_string(), FunctionSignature { params: vec![("s".into(), Type::String, false)], return_type: Some(Type::String), is_async: false });
        env.define_function("to_upper".to_string(), FunctionSignature { params: vec![("s".into(), Type::String, false)], return_type: Some(Type::String), is_async: false });
        env.define_function("to_lower".to_string(), FunctionSignature { params: vec![("s".into(), Type::String, false)], return_type: Some(Type::String), is_async: false });
        env.define_function("starts_with".to_string(), FunctionSignature { params: vec![("s".into(), Type::String, false), ("prefix".into(), Type::String, false)], return_type: Some(Type::Bool), is_async: false });
        env.define_function("ends_with".to_string(), FunctionSignature { params: vec![("s".into(), Type::String, false), ("suffix".into(), Type::String, false)], return_type: Some(Type::Bool), is_async: false });
        env.define_function("parse_int".to_string(), FunctionSignature { params: vec![("s".into(), Type::String, false)], return_type: Some(Type::Option(Box::new(Type::Int64))), is_async: false });
        env.define_function("parse_float".to_string(), FunctionSignature { params: vec![("s".into(), Type::String, false)], return_type: Some(Type::Option(Box::new(Type::Float64))), is_async: false });
        env.define_function("int_to_string".to_string(), FunctionSignature { params: vec![("value".into(), Type::Int64, false)], return_type: Some(Type::String), is_async: false });
        env.define_function("float_to_string".to_string(), FunctionSignature { params: vec![("value".into(), Type::Float64, false)], return_type: Some(Type::String), is_async: false });
        env.define_function("bool_to_string".to_string(), FunctionSignature { params: vec![("value".into(), Type::Bool, false)], return_type: Some(Type::String), is_async: false });
        env.define_function("char_at".to_string(), FunctionSignature { params: vec![("s".into(), Type::String, false), ("index".into(), Type::Int64, false)], return_type: Some(Type::String), is_async: false });

        // std.fs
        env.define_function("file_read".to_string(), FunctionSignature { params: vec![("path".into(), Type::String, false)], return_type: Some(Type::String), is_async: false });
        env.define_function("file_write".to_string(), FunctionSignature { params: vec![("path".into(), Type::String, false), ("content".into(), Type::String, false)], return_type: Some(Type::Bool), is_async: false });
        env.define_function("file_append".to_string(), FunctionSignature { params: vec![("path".into(), Type::String, false), ("content".into(), Type::String, false)], return_type: Some(Type::Bool), is_async: false });
        env.define_function("file_exists".to_string(), FunctionSignature { params: vec![("path".into(), Type::String, false)], return_type: Some(Type::Bool), is_async: false });
        env.define_function("create_dir".to_string(), FunctionSignature { params: vec![("path".into(), Type::String, false)], return_type: Some(Type::Bool), is_async: false });
        env.define_function("remove_file".to_string(), FunctionSignature { params: vec![("path".into(), Type::String, false)], return_type: Some(Type::Bool), is_async: false });
        env.define_function("list_dir".to_string(), FunctionSignature { params: vec![("path".into(), Type::String, false)], return_type: Some(Type::String), is_async: false });
        env.define_function("file_copy".to_string(), FunctionSignature { params: vec![("src".into(), Type::String, false), ("dst".into(), Type::String, false)], return_type: Some(Type::Bool), is_async: false });
        env.define_function("file_rename".to_string(), FunctionSignature { params: vec![("old".into(), Type::String, false), ("new".into(), Type::String, false)], return_type: Some(Type::Bool), is_async: false });

        // std.mem
        env.define_function("alloc".to_string(), FunctionSignature { params: vec![("size".into(), Type::Int64, false)], return_type: Some(Type::Bytes), is_async: false });
        env.define_function("free".to_string(), FunctionSignature { params: vec![("ptr".into(), Type::Bytes, false)], return_type: Some(Type::Void), is_async: false });
        env.define_function("memcpy".to_string(), FunctionSignature { params: vec![("dst".into(), Type::Bytes, false), ("src".into(), Type::Bytes, false), ("size".into(), Type::Int64, false)], return_type: Some(Type::Void), is_async: false });
        env.define_function("memset".to_string(), FunctionSignature { params: vec![("dst".into(), Type::Bytes, false), ("value".into(), Type::Int64, false), ("size".into(), Type::Int64, false)], return_type: Some(Type::Void), is_async: false });

        // std.conv
        env.define_function("to_int".to_string(), FunctionSignature { params: vec![("s".into(), Type::String, false)], return_type: Some(Type::Option(Box::new(Type::Int64))), is_async: false });
        env.define_function("to_float".to_string(), FunctionSignature { params: vec![("s".into(), Type::String, false)], return_type: Some(Type::Option(Box::new(Type::Float64))), is_async: false });
        env.define_function("to_bool".to_string(), FunctionSignature { params: vec![("s".into(), Type::String, false)], return_type: Some(Type::Option(Box::new(Type::Bool))), is_async: false });

        // std.assert
        env.define_function("assert".to_string(), FunctionSignature { params: vec![("condition".into(), Type::Bool, false), ("msg".into(), Type::String, false)], return_type: Some(Type::Void), is_async: false });
        env.define_function("assert_eq".to_string(), FunctionSignature { params: vec![("a".into(), Type::String, false), ("b".into(), Type::String, false), ("msg".into(), Type::String, false)], return_type: Some(Type::Void), is_async: false });
        env.define_function("assert_ne".to_string(), FunctionSignature { params: vec![("a".into(), Type::String, false), ("b".into(), Type::String, false), ("msg".into(), Type::String, false)], return_type: Some(Type::Void), is_async: false });
        env.define_function("assert_true".to_string(), FunctionSignature { params: vec![("value".into(), Type::Bool, false), ("msg".into(), Type::String, false)], return_type: Some(Type::Void), is_async: false });
        env.define_function("assert_false".to_string(), FunctionSignature { params: vec![("value".into(), Type::Bool, false), ("msg".into(), Type::String, false)], return_type: Some(Type::Void), is_async: false });

        TypeChecker { env }
    }

    pub fn check_program(&mut self, program: &Program) -> Result<(), TypeError> {
        // First pass: collect all type definitions and function signatures
        for item in &program.items {
            match item {
                TopLevelItem::Struct { name, fields, .. } => {
                    self.env.define_type(
                        name.clone(),
                        TypeDefinition::Struct {
                            fields: fields.clone(),
                        },
                    );
                }
                TopLevelItem::Enum { name, variants, .. } => {
                    let enum_variants = variants
                        .iter()
                        .map(|v| EnumVariantInfo {
                            name: v.name.clone(),
                            data: v.data.clone(),
                        })
                        .collect();
                    self.env.define_type(
                        name.clone(),
                        TypeDefinition::Enum {
                            variants: enum_variants,
                        },
                    );
                }
                TopLevelItem::Function {
                    name,
                    params,
                    return_type,
                    is_async,
                    ..
                } => {
                    let sig = FunctionSignature {
                        params: params
                            .iter()
                            .map(|p| (p.name.clone(), p.ty.clone(), p.is_move))
                            .collect(),
                        return_type: return_type.clone(),
                        is_async: *is_async,
                    };
                    self.env.define_function(name.clone(), sig);
                }
                TopLevelItem::Impl {
                    type_name,
                    methods,
                    ..
                } => {
                    // Register impl methods as functions
                    for method in methods {
                        let sig = FunctionSignature {
                            params: method.params
                                .iter()
                                .map(|p| (p.name.clone(), p.ty.clone(), p.is_move))
                                .collect(),
                            return_type: method.return_type.clone(),
                            is_async: method.is_async,
                        };
                        // Method name: Type.Method
                        let full_name = format!("{}.{}", type_name, method.name);
                        self.env.define_function(full_name, sig);
                    }
                }
                TopLevelItem::Const { name, ty, value, .. } => {
                    let value_type = self.check_expression(value)?;
                    if !self.types_compatible(ty, &value_type) {
                        return Err(TypeError::ConstTypeMismatch {
                            expected: format!("{:?}", ty),
                            found: format!("{:?}", value_type),
                        });
                    }
                    self.env.define_constant(name.clone(), ty.clone());
                    self.env.define_variable(name.clone(), ty.clone());
                }
                _ => {}
            }
        }

        // Second pass: check each item
        for item in &program.items {
            self.check_top_level_item(item)?;
        }

        Ok(())
    }

    fn check_top_level_item(&mut self, item: &TopLevelItem) -> Result<(), TypeError> {
        match item {
            TopLevelItem::Function {
                params,
                body,
                return_type,
                ..
            } => {
                // Create new scope for function
                let mut func_env = self.env.clone();

                // Add parameters to scope
                for param in params {
                    func_env.define_variable(param.name.clone(), param.ty.clone());
                }

                // Check body
                let mut checker = TypeChecker { env: func_env };
                checker.check_block(body, return_type.as_ref())?;

                Ok(())
            }
            TopLevelItem::Impl {
                type_name,
                methods,
                ..
            } => {
                for method in methods {
                    let mut func_env = self.env.clone();
                    for param in &method.params {
                        func_env.define_variable(param.name.clone(), param.ty.clone());
                    }
                    let mut checker = TypeChecker { env: func_env };
                    checker.check_block(&method.body, method.return_type.as_ref())?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn check_block(
        &mut self,
        stmts: &[Stmt],
        _expected_return: Option<&Type>,
    ) -> Result<Option<Type>, TypeError> {
        for (i, stmt) in stmts.iter().enumerate() {
            let is_last = i == stmts.len() - 1;
            let result = self.check_statement(stmt)?;

            if is_last {
                return Ok(result);
            }
        }

        Ok(None)
    }

    fn check_statement(&mut self, stmt: &Stmt) -> Result<Option<Type>, TypeError> {
        match stmt {
            Stmt::Let {
                name,
                ty,
                value,
                mutable: _,
            } => {
                let value_type = self.check_expression(value)?;

                if let Some(expected) = ty {
                    if !self.types_compatible(expected, &value_type) {
                        return Err(TypeError::TypeMismatch {
                            expected: format!("{:?}", expected),
                            found: format!("{:?}", value_type),
                        });
                    }
                }

                self.env.define_variable(name.clone(), value_type);
                Ok(None)
            }
            Stmt::Assignment { target, value } => {
                let target_type = self.check_expression(target)?;
                let value_type = self.check_expression(value)?;

                if !self.types_compatible(&target_type, &value_type) {
                    return Err(TypeError::TypeMismatch {
                        expected: format!("{:?}", target_type),
                        found: format!("{:?}", value_type),
                    });
                }

                Ok(None)
            }
            Stmt::Expression(expr) => {
                let ty = self.check_expression(expr)?;
                Ok(Some(ty))
            }
            Stmt::Return(expr) => {
                if let Some(e) = expr {
                    let ty = self.check_expression(e)?;
                    Ok(Some(ty))
                } else {
                    Ok(None)
                }
            }
            Stmt::Break | Stmt::Continue => Ok(None),
            Stmt::Loop(body) => {
                self.check_block(body, None)?;
                Ok(None)
            }
            Stmt::While { condition, body } => {
                let cond_type = self.check_expression(condition)?;
                if !matches!(cond_type, Type::Bool) {
                    return Err(TypeError::WhileConditionNotBool);
                }
                self.check_block(body, None)?;
                Ok(None)
            }
            Stmt::For {
                variable,
                iterable,
                body,
            } => {
                let iter_type = self.check_expression(iterable)?;

                let elem_type = match iter_type {
                    Type::List(elem) => *elem,
                    Type::Map(key, _value) => *key,
                    Type::Array(elem, _) => *elem,
                    _ => return Err(TypeError::InvalidIterableType),
                };

                self.env.define_variable(variable.clone(), elem_type);
                self.check_block(body, None)?;
                Ok(None)
            }
            Stmt::Guard { condition, else_body } => {
                let cond_type = self.check_expression(condition)?;
                if !matches!(cond_type, Type::Bool) {
                    return Err(TypeError::GuardConditionNotBool);
                }
                self.check_block(else_body, None)?;
                Ok(None)
            }
            Stmt::Spawn(expr) => {
                let ty = self.check_expression(expr)?;
                match ty {
                    Type::Async(_) => Ok(None),
                    _ => Err(TypeError::TypeMismatch {
                        expected: "Async<T>".to_string(),
                        found: format!("{:?}", ty),
                    }),
                }
            }
        }
    }

    fn check_expression(&mut self, expr: &Expr) -> Result<Type, TypeError> {
        match expr {
            Expr::IntegerLiteral(_) => Ok(Type::Int64),
            Expr::FloatLiteral(_) => Ok(Type::Float64),
            Expr::StringLiteral(_) => Ok(Type::String),
            Expr::BoolLiteral(_) => Ok(Type::Bool),
            Expr::HexLiteral(_) => Ok(Type::Int64),

            Expr::Identifier(name) => {
                // Check variables first, then constants
                if let Some(ty) = self.env.get_variable(name) {
                    return Ok(ty.clone());
                }
                if let Some(ty) = self.env.get_constant(name) {
                    return Ok(ty.clone());
                }
                Err(TypeError::UndefinedVariable(name.clone()))
            }

            Expr::BinaryOp { op, left, right } => {
                let left_type = self.check_expression(left)?;
                let right_type = self.check_expression(right)?;

                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        match (&left_type, &right_type) {
                            (Type::Int64, Type::Int64) => Ok(Type::Int64),
                            (Type::Float64, Type::Float64) => Ok(Type::Float64),
                            (Type::UInt64, Type::UInt64) => Ok(Type::UInt64),
                            _ => Err(TypeError::CannotApplyOperator(format!("{:?}", op))),
                        }
                    }
                    BinOp::Concat => {
                        match (&left_type, &right_type) {
                            (Type::String, Type::String) => Ok(Type::String),
                            (Type::String, Type::Int64) => Ok(Type::String),
                            (Type::String, Type::Float64) => Ok(Type::String),
                            (Type::String, Type::UInt64) => Ok(Type::String),
                            (Type::String, Type::Bool) => Ok(Type::String),
                            (Type::Int64, Type::String) => Ok(Type::String),
                            (Type::Float64, Type::String) => Ok(Type::String),
                            (Type::UInt64, Type::String) => Ok(Type::String),
                            (Type::Bool, Type::String) => Ok(Type::String),
                            (Type::List(a), Type::List(b)) => {
                                if a == b {
                                    Ok(Type::List(a.clone()))
                                } else {
                                    Err(TypeError::TypeMismatch {
                                        expected: format!("{:?}", a),
                                        found: format!("{:?}", b),
                                    })
                                }
                            }
                            _ => Err(TypeError::CannotApplyOperator("Concat".to_string())),
                        }
                    }
                    BinOp::Eq | BinOp::Neq => {
                        if self.types_compatible(&left_type, &right_type) {
                            Ok(Type::Bool)
                        } else {
                            Err(TypeError::TypeMismatch {
                                expected: format!("{:?}", left_type),
                                found: format!("{:?}", right_type),
                            })
                        }
                    }
                    BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                        match (&left_type, &right_type) {
                            (Type::Int64, Type::Int64) => Ok(Type::Bool),
                            (Type::Float64, Type::Float64) => Ok(Type::Bool),
                            (Type::UInt64, Type::UInt64) => Ok(Type::Bool),
                            (Type::String, Type::String) => Ok(Type::Bool),
                            _ => Err(TypeError::CannotApplyOperator(format!("{:?}", op))),
                        }
                    }
                    BinOp::And | BinOp::Or => {
                        if matches!(left_type, Type::Bool) && matches!(right_type, Type::Bool) {
                            Ok(Type::Bool)
                        } else {
                            Err(TypeError::CannotApplyOperator(format!("{:?}", op)))
                        }
                    }
                }
            }

            Expr::UnaryOp { op, expr } => {
                let expr_type = self.check_expression(expr)?;
                match op {
                    UnaryOp::Neg => match expr_type {
                        Type::Int64 | Type::Float64 | Type::UInt64 => Ok(expr_type),
                        _ => Err(TypeError::CannotApplyOperator("Neg".to_string())),
                    },
                    UnaryOp::Not => {
                        if matches!(expr_type, Type::Bool) {
                            Ok(Type::Bool)
                        } else {
                            Err(TypeError::CannotApplyOperator("Not".to_string()))
                        }
                    }
                }
            }

            Expr::Ref(expr) => {
                let expr_type = self.check_expression(expr)?;
                Ok(Type::Ref(Box::new(expr_type)))
            }

            Expr::Cast { expr, target_type } => {
                let source_type = self.check_expression(expr)?;
                // Allow casting between numeric types and to/from String
                match (&source_type, target_type) {
                    (Type::Int64, Type::Int64)
                    | (Type::Int64, Type::UInt64)
                    | (Type::Int64, Type::Float64)
                    | (Type::UInt64, Type::Int64)
                    | (Type::UInt64, Type::UInt64)
                    | (Type::UInt64, Type::Float64)
                    | (Type::Float64, Type::Int64)
                    | (Type::Float64, Type::UInt64)
                    | (Type::Float64, Type::Float64)
                    | (Type::String, Type::Int64)
                    | (Type::Int64, Type::String)
                    | (Type::String, Type::Float64)
                    | (Type::Float64, Type::String) => Ok(target_type.clone()),
                    _ => Err(TypeError::InvalidCast {
                        from: format!("{:?}", source_type),
                        to: format!("{:?}", target_type),
                    }),
                }
            }

            Expr::FunctionCall { name, args } => {
                let func_name = match name.as_ref() {
                    Expr::Identifier(n) => {
                        // Convert qualified name: math::add -> math_add
                        n.replace("::", "_")
                    }
                    _ => return Err(TypeError::CannotCallNonFunction("non-identifier".to_string())),
                };

                let sig = self
                    .env
                    .get_function(&func_name)
                    .ok_or_else(|| TypeError::UndefinedFunction(func_name.clone()))?
                    .clone();

                if args.len() != sig.params.len() {
                    return Err(TypeError::WrongArgumentCount {
                        expected: sig.params.len(),
                        found: args.len(),
                    });
                }

                for (arg, (_, param_type, _)) in args.iter().zip(&sig.params) {
                    let arg_type = self.check_expression(arg)?;
                    if !self.types_compatible(param_type, &arg_type) {
                        return Err(TypeError::TypeMismatch {
                            expected: format!("{:?}", param_type),
                            found: format!("{:?}", arg_type),
                        });
                    }
                }

                Ok(sig.return_type.unwrap_or(Type::Void))
            }

            Expr::MethodCall {
                object,
                method,
                args,
            } => {
                let object_type = self.check_expression(object)?;

                match &object_type {
                    Type::String => match method.as_str() {
                        "len" => Ok(Type::UInt64),
                        "to_uint" => Ok(Type::Option(Box::new(Type::UInt64))),
                        _ => Err(TypeError::UndefinedFunction(format!("String.{}", method))),
                    },
                    Type::List(elem_type) => match method.as_str() {
                        "len" => Ok(Type::UInt64),
                        "iter" => Ok(Type::Async(elem_type.clone())),
                        "append" => {
                            if args.len() == 1 {
                                let arg_type = self.check_expression(&args[0])?;
                                if self.types_compatible(elem_type, &arg_type) {
                                    Ok(Type::Void)
                                } else {
                                    Err(TypeError::TypeMismatch {
                                        expected: format!("{:?}", elem_type),
                                        found: format!("{:?}", arg_type),
                                    })
                                }
                            } else {
                                Err(TypeError::WrongArgumentCount {
                                    expected: 1,
                                    found: args.len(),
                                })
                            }
                        }
                        _ => Err(TypeError::UndefinedFunction(format!("List.{}", method))),
                    },
                    Type::Channel(elem_type) => match method.as_str() {
                        "send" => {
                            if args.len() == 1 {
                                let arg_type = self.check_expression(&args[0])?;
                                if self.types_compatible(elem_type, &arg_type) {
                                    Ok(Type::Option(Box::new(Type::Void)))
                                } else {
                                    Err(TypeError::TypeMismatch {
                                        expected: format!("{:?}", elem_type),
                                        found: format!("{:?}", arg_type),
                                    })
                                }
                            } else {
                                Err(TypeError::WrongArgumentCount {
                                    expected: 1,
                                    found: args.len(),
                                })
                            }
                        }
                        "recv" => Ok(Type::Option(elem_type.clone())),
                        _ => Err(TypeError::UndefinedFunction(format!("Channel.{}", method))),
                    },
                    Type::Custom(type_name) => {
                        // Check if method exists in impl blocks
                        let full_name = format!("{}.{}", type_name, method);
                        if let Some(sig) = self.env.get_function(&full_name).cloned() {
                            if args.len() != sig.params.len() {
                                return Err(TypeError::WrongArgumentCount {
                                    expected: sig.params.len(),
                                    found: args.len(),
                                });
                            }
                            for (arg, (_, param_type, _)) in args.iter().zip(&sig.params) {
                                let arg_type = self.check_expression(arg)?;
                                if !self.types_compatible(param_type, &arg_type) {
                                    return Err(TypeError::TypeMismatch {
                                        expected: format!("{:?}", param_type),
                                        found: format!("{:?}", arg_type),
                                    });
                                }
                            }
                            Ok(sig.return_type.unwrap_or(Type::Void))
                        } else {
                            Err(TypeError::UndefinedFunction(format!("{}.{}", type_name, method)))
                        }
                    }
                    _ => Err(TypeError::CannotCallNonFunction(format!("{:?}", object_type))),
                }
            }

            Expr::FieldAccess { object, field } => {
                let object_type = self.check_expression(object)?;

                // Unwrap reference type if needed
                let actual_type = match &object_type {
                    Type::Ref(inner) => (**inner).clone(),
                    _ => object_type.clone(),
                };

                match &actual_type {
                    Type::Custom(type_name) => {
                        let type_def = self
                            .env
                            .get_type(type_name)
                            .ok_or_else(|| TypeError::UndefinedType(type_name.clone()))?
                            .clone();

                        match type_def {
                            TypeDefinition::Struct { fields } => {
                                for (field_name, field_type) in &fields {
                                    if field_name == field {
                                        return Ok(field_type.clone());
                                    }
                                }
                                Err(TypeError::UndefinedVariable(format!(
                                    "{}.{}",
                                    type_name, field
                                )))
                            }
                            _ => Err(TypeError::UndefinedVariable(format!(
                                "{}.{}",
                                type_name, field
                            ))),
                        }
                    }
                    _ => Err(TypeError::CannotApplyOperator("FieldAccess".to_string())),
                }
            }

            Expr::IndexAccess { object, index } => {
                let object_type = self.check_expression(object)?;
                let index_type = self.check_expression(index)?;

                match &object_type {
                    Type::List(elem_type) => {
                        if matches!(index_type, Type::UInt64 | Type::Int64) {
                            Ok((**elem_type).clone())
                        } else {
                            Err(TypeError::TypeMismatch {
                                expected: "UInt64 or Int64".to_string(),
                                found: format!("{:?}", index_type),
                            })
                        }
                    }
                    Type::Map(key_type, value_type) => {
                        if self.types_compatible(&key_type, &index_type) {
                            Ok((**value_type).clone())
                        } else {
                            Err(TypeError::TypeMismatch {
                                expected: format!("{:?}", key_type),
                                found: format!("{:?}", index_type),
                            })
                        }
                    }
                    Type::Array(elem_type, _) => {
                        if matches!(index_type, Type::UInt64 | Type::Int64) {
                            Ok((**elem_type).clone())
                        } else {
                            Err(TypeError::TypeMismatch {
                                expected: "UInt64 or Int64".to_string(),
                                found: format!("{:?}", index_type),
                            })
                        }
                    }
                    _ => Err(TypeError::CannotApplyOperator("IndexAccess".to_string())),
                }
            }

            Expr::Range { start, end, .. } => {
                let start_type = self.check_expression(start)?;
                let end_type = self.check_expression(end)?;
                if self.types_compatible(&start_type, &end_type) {
                    Ok(Type::List(Box::new(start_type)))
                } else {
                    Err(TypeError::TypeMismatch {
                        expected: format!("{:?}", start_type),
                        found: format!("{:?}", end_type),
                    })
                }
            }

            Expr::ArrayLiteral(elements) => {
                if elements.is_empty() {
                    return Ok(Type::List(Box::new(Type::Void)));
                }
                let first_type = self.check_expression(&elements[0])?;
                for elem in &elements[1..] {
                    let elem_type = self.check_expression(elem)?;
                    if !self.types_compatible(&first_type, &elem_type) {
                        return Err(TypeError::ArrayTypeMismatch {
                            expected: format!("{:?}", first_type),
                            found: format!("{:?}", elem_type),
                        });
                    }
                }
                Ok(Type::List(Box::new(first_type)))
            }

            Expr::Match { expr, arms } => {
                let expr_type = self.check_expression(expr)?;

                // Unwrap reference type for pattern matching
                let match_type = match &expr_type {
                    Type::Ref(inner) => (**inner).clone(),
                    _ => expr_type.clone(),
                };

                for arm in arms {
                    let mut arm_env = self.env.clone();
                    self.check_pattern(&arm.pattern, &match_type, &mut arm_env)?;

                    // Check guard if present
                    if let Some(guard) = &arm.guard {
                        let mut checker = TypeChecker { env: arm_env.clone() };
                        let guard_type = checker.check_expression(guard)?;
                        if !matches!(guard_type, Type::Bool) {
                            return Err(TypeError::GuardConditionNotBool);
                        }
                    }

                    let mut checker = TypeChecker { env: arm_env };
                    checker.check_expression(&arm.body)?;
                }

                // TODO: Return common type of all arms
                Ok(Type::Void)
            }

            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_type = self.check_expression(condition)?;
                if !matches!(cond_type, Type::Bool) {
                    return Err(TypeError::TypeMismatch {
                        expected: "Bool".to_string(),
                        found: format!("{:?}", cond_type),
                    });
                }

                let then_type = self.check_expression(then_branch)?;

                if let Some(else_expr) = else_branch {
                    let _else_type = self.check_expression(else_expr)?;
                    // TODO: Find common type
                    Ok(then_type)
                } else {
                    Ok(Type::Void)
                }
            }

            Expr::Block(stmts) => {
                let block_env = self.env.clone();
                let mut checker = TypeChecker { env: block_env };

                for (i, stmt) in stmts.iter().enumerate() {
                    let is_last = i == stmts.len() - 1;
                    let result = checker.check_statement(stmt)?;

                    if is_last {
                        return Ok(result.unwrap_or(Type::Void));
                    }
                }

                Ok(Type::Void)
            }

            Expr::ChannelBounded { capacity } => {
                let cap_type = self.check_expression(capacity)?;
                if !matches!(cap_type, Type::UInt64 | Type::Int64) {
                    return Err(TypeError::TypeMismatch {
                        expected: "UInt64 or Int64".to_string(),
                        found: format!("{:?}", cap_type),
                    });
                }
                Ok(Type::Channel(Box::new(Type::String))) // Default to Channel<String>
            }

            Expr::StructInit { name, fields } => {
                // Check if the type exists
                let type_def = self
                    .env
                    .get_type(name)
                    .ok_or_else(|| TypeError::UndefinedType(name.clone()))?
                    .clone();

                match type_def {
                    TypeDefinition::Struct { fields: struct_fields } => {
                        // Check each field
                        for (field_name, field_value) in fields {
                            let field_type = self.check_expression(field_value)?;
                            let expected_type = struct_fields
                                .iter()
                                .find(|(n, _)| n == field_name)
                                .map(|(_, t)| t)
                                .ok_or_else(|| TypeError::UndefinedVariable(format!("{}.{}", name, field_name)))?;

                            if !self.types_compatible(expected_type, &field_type) {
                                return Err(TypeError::TypeMismatch {
                                    expected: format!("{:?}", expected_type),
                                    found: format!("{:?}", field_type),
                                });
                            }
                        }
                        Ok(Type::Custom(name.clone()))
                    }
                    _ => Err(TypeError::TypeMismatch {
                        expected: "Struct type".to_string(),
                        found: format!("{:?}", name),
                    }),
                }
            }

            Expr::EnumInit { enum_name, variant, args } => {
                // Check if the enum type exists
                let type_def = self
                    .env
                    .get_type(enum_name)
                    .ok_or_else(|| TypeError::UndefinedType(enum_name.clone()))?
                    .clone();

                match type_def {
                    TypeDefinition::Enum { variants } => {
                        // Find the variant
                        let variant_def = variants
                            .iter()
                            .find(|v| v.name == *variant)
                            .ok_or_else(|| TypeError::UndefinedVariable(format!("{}::{}", enum_name, variant)))?;

                        // Check args count
                        let expected_data = variant_def.data.as_ref().map_or(0, |d| d.len());
                        if args.len() != expected_data {
                            return Err(TypeError::WrongArgumentCount {
                                expected: expected_data,
                                found: args.len(),
                            });
                        }

                        // Check each arg type
                        if let Some(expected_types) = &variant_def.data {
                            for (arg, expected_type) in args.iter().zip(expected_types) {
                                let arg_type = self.check_expression(arg)?;
                                if !self.types_compatible(expected_type, &arg_type) {
                                    return Err(TypeError::TypeMismatch {
                                        expected: format!("{:?}", expected_type),
                                        found: format!("{:?}", arg_type),
                                    });
                                }
                            }
                        }

                        Ok(Type::Custom(enum_name.clone()))
                    }
                    _ => Err(TypeError::TypeMismatch {
                        expected: "Enum type".to_string(),
                        found: format!("{:?}", enum_name),
                    }),
                }
            }

            Expr::Await(expr) => {
                let expr_type = self.check_expression(expr)?;
                match expr_type {
                    Type::Async(inner) => Ok(*inner),
                    _ => Err(TypeError::TypeMismatch {
                        expected: "Async<T>".to_string(),
                        found: format!("{:?}", expr_type),
                    }),
                }
            }
        }
    }

    fn check_pattern(
        &self,
        pattern: &Pattern,
        expected_type: &Type,
        env: &mut TypeEnv,
    ) -> Result<(), TypeError> {
        match pattern {
            Pattern::IntegerLiteral(_) => {
                if matches!(expected_type, Type::Int64 | Type::UInt64) {
                    Ok(())
                } else {
                    Err(TypeError::TypeMismatch {
                        expected: "Int64 or UInt64".to_string(),
                        found: format!("{:?}", expected_type),
                    })
                }
            }
            Pattern::StringLiteral(_) => {
                if matches!(expected_type, Type::String) {
                    Ok(())
                } else {
                    Err(TypeError::TypeMismatch {
                        expected: "String".to_string(),
                        found: format!("{:?}", expected_type),
                    })
                }
            }
            Pattern::BoolLiteral(_) => {
                if matches!(expected_type, Type::Bool) {
                    Ok(())
                } else {
                    Err(TypeError::TypeMismatch {
                        expected: "Bool".to_string(),
                        found: format!("{:?}", expected_type),
                    })
                }
            }
            Pattern::Identifier(name) => {
                env.define_variable(name.clone(), expected_type.clone());
                Ok(())
            }
            Pattern::EnumVariant { enum_name: _, variant, data } => {
                // Check if variant matches expected type
                match expected_type {
                    Type::Custom(type_name) => {
                        let type_def = env.get_type(type_name).cloned();
                        if let Some(TypeDefinition::Enum { variants }) = type_def {
                            for v in variants {
                                if v.name == *variant {
                                    // Check data patterns if present
                                    if let Some(data_patterns) = data {
                                        if let Some(expected_data) = &v.data {
                                            if data_patterns.len() != expected_data.len() {
                                                return Err(TypeError::WrongArgumentCount {
                                                    expected: expected_data.len(),
                                                    found: data_patterns.len(),
                                                });
                                            }
                                            // Recursively check each data pattern
                                            for (dp, dt) in data_patterns.iter().zip(expected_data) {
                                                self.check_pattern(dp, dt, env)?;
                                            }
                                        }
                                    }
                                    return Ok(());
                                }
                            }
                        }
                        Err(TypeError::UndefinedVariable(format!(
                            "{}::{}",
                            type_name, variant
                        )))
                    }
                    _ => Err(TypeError::TypeMismatch {
                        expected: "Enum type".to_string(),
                        found: format!("{:?}", expected_type),
                    }),
                }
            }
            Pattern::Wildcard => Ok(()),
        }
    }

    fn types_compatible(&self, a: &Type, b: &Type) -> bool {
        match (a, b) {
            (Type::String, Type::String) => true,
            (Type::UInt64, Type::UInt64) => true,
            (Type::Int64, Type::Int64) => true,
            (Type::Float64, Type::Float64) => true,
            (Type::Bool, Type::Bool) => true,
            (Type::Void, Type::Void) => true,
            (Type::Bytes, Type::Bytes) => true,
            (Type::List(a), Type::List(b)) => self.types_compatible(a, b),
            (Type::Map(k1, v1), Type::Map(k2, v2)) => {
                self.types_compatible(k1, k2) && self.types_compatible(v1, v2)
            }
            (Type::Result(o1, e1), Type::Result(o2, e2)) => {
                self.types_compatible(o1, o2) && self.types_compatible(e1, e2)
            }
            (Type::Option(a), Type::Option(b)) => self.types_compatible(a, b),
            (Type::Async(a), Type::Async(b)) => self.types_compatible(a, b),
            (Type::Channel(a), Type::Channel(b)) => self.types_compatible(a, b),
            (Type::Ref(a), Type::Ref(b)) => self.types_compatible(a, b),
            (Type::Custom(a), Type::Custom(b)) => a == b,
            (Type::Array(a, s1), Type::Array(b, s2)) => self.types_compatible(a, b) && s1 == s2,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    #[test]
    fn test_type_check_function() {
        let input = "@fn add(a: Int64, b: Int64) -> Int64 { return a + b; }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_type_check_struct() {
        let input = "@struct Point { x: Float64, y: Float64 }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_type_check_while() {
        let input = "@fn count() -> Void { let i: Int64 = 0; while i < 10 { i = i + 1; } }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }
}
