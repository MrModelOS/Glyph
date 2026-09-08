use crate::ast::*;
use crate::generics::{infer_with_expected, instance_c_name, substitute_params, substitute_stmt, substitute_type, GenericFnInfo};
use std::collections::{HashMap, HashSet};
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

    #[error("Guard condition must be Bool")]
    GuardConditionNotBool,

    #[error("For loop iterable must be a List or a range")]
    InvalidIterableType,

    #[error("Invalid cast from {from} to {to}")]
    InvalidCast { from: String, to: String },

    #[error("While condition must be Bool")]
    WhileConditionNotBool,

    #[error("Array type mismatch: expected {expected}, found {found}")]
    ArrayTypeMismatch { expected: String, found: String },

    #[error("Map value type mismatch: expected {expected}, found {found}")]
    MappingValueMismatch { expected: String, found: String },

    #[error("Const type mismatch: expected {expected}, found {found}")]
    ConstTypeMismatch { expected: String, found: String },

    #[error("Test function '{name}' must not have parameters")]
    TestFunctionParams { name: String },

    #[error("Test function '{name}' must return Void (found {found})")]
    TestFunctionReturn { name: String, found: String },

    #[error("Cannot infer type arguments for generic function '{0}'")]
    CannotInferTypeArgs(String),

    #[error("Async generic functions are not supported: '{0}'")]
    AsyncGeneric(String),

    #[error("in function '{name}': {source}")]
    InFunction {
        name: String,
        source: Box<TypeError>,
    },

    #[error("line {loc}: {source}")]
    AtLine {
        loc: LineCol,
        source: Box<TypeError>,
    },
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
    generic_fns: HashMap<String, GenericFnInfo>,
    instantiated: HashSet<String>,
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

        // test framework
        env.define_function("__builtin_assert".to_string(), FunctionSignature { params: vec![("condition".into(), Type::Bool, false), ("msg".into(), Type::String, false)], return_type: Some(Type::Void), is_async: false });
        env.define_function("__builtin_assert_true".to_string(), FunctionSignature { params: vec![("value".into(), Type::Bool, false), ("msg".into(), Type::String, false)], return_type: Some(Type::Void), is_async: false });
        env.define_function("__builtin_assert_false".to_string(), FunctionSignature { params: vec![("value".into(), Type::Bool, false), ("msg".into(), Type::String, false)], return_type: Some(Type::Void), is_async: false });

        TypeChecker {
            env,
            generic_fns: HashMap::new(),
            instantiated: HashSet::new(),
        }
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
                    type_params,
                    params,
                    return_type,
                    body,
                    is_async,
                    is_test,
                    ..
                } => {
                    if !type_params.is_empty() {
                        if *is_async {
                            return Err(TypeError::AsyncGeneric(name.clone()));
                        }
                        self.generic_fns.insert(
                            name.clone(),
                            GenericFnInfo {
                                type_params: type_params.clone(),
                                params: params.clone(),
                                return_type: return_type.clone(),
                                body: body.clone(),
                            },
                        );
                        continue;
                    }
                    if *is_test {
                        if !params.is_empty() {
                            return Err(TypeError::TestFunctionParams {
                                name: name.clone(),
                            });
                        }
                        match return_type {
                            Some(Type::Void) | None => {}
                            Some(other) => {
                                return Err(TypeError::TestFunctionReturn {
                                    name: name.clone(),
                                    found: format!("{}", other),
                                });
                            }
                        }
                    }
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
                    if !self.types_compatible(ty, &value_type)
                        && !self.int_literal_satisfies(ty, &value_type, value)
                    {
                        return Err(TypeError::ConstTypeMismatch {
                            expected: format!("{}", ty),
                            found: format!("{}", value_type),
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
                name,
                type_params,
                params,
                body,
                return_type,
                ..
            } => {
                if !type_params.is_empty() {
                    return Ok(());
                }
                // Create new scope for function
                let mut func_env = self.env.clone();

                // Add parameters to scope
                for param in params {
                    func_env.define_variable(param.name.clone(), param.ty.clone());
                }

                // Check body
                let mut checker = TypeChecker {
                    env: func_env,
                    generic_fns: self.generic_fns.clone(),
                    instantiated: self.instantiated.clone(),
                };
                checker
                    .check_block(body, return_type.as_ref())
                    .map_err(|e| TypeError::InFunction {
                        name: name.clone(),
                        source: Box::new(e),
                    })?;
                self.instantiated = checker.instantiated;

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
                    let mut checker = TypeChecker {
                        env: func_env,
                        generic_fns: self.generic_fns.clone(),
                        instantiated: self.instantiated.clone(),
                    };
                    checker
                        .check_block(&method.body, method.return_type.as_ref())
                        .map_err(|e| TypeError::InFunction {
                            name: format!("{}::{}", type_name, method.name),
                            source: Box::new(e),
                        })?;
                    self.instantiated = checker.instantiated;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Names of assertion builtins that accept polymorphic values
    fn is_polymorphic_assert(name: &str) -> bool {
        matches!(
            name,
            "assert"
                | "assert_eq"
                | "assert_ne"
                | "assert_true"
                | "assert_false"
                | "__builtin_assert"
                | "__builtin_assert_eq"
                | "__builtin_assert_ne"
                | "__builtin_assert_true"
                | "__builtin_assert_false"
        )
    }

    fn check_assert_builtin_call(
        &mut self,
        name: &str,
        args: &[Expr],
    ) -> Result<Type, TypeError> {
        let is_eq = name.ends_with("assert_eq") || name.ends_with("assert_ne");

        // assert/assert_true/assert_false: (Bool, String)
        // assert_eq/assert_ne: (T, T, String)
        let expected_args = if is_eq { 3 } else { 2 };
        if args.len() != expected_args {
            return Err(TypeError::WrongArgumentCount {
                expected: expected_args,
                found: args.len(),
            });
        }

        // Check message string
        let msg_type = self.check_expression(&args[if is_eq { 2 } else { 1 }])?;
        if !self.types_compatible(&Type::String, &msg_type) {
            return Err(TypeError::TypeMismatch {
                expected: "String".to_string(),
                found: format!("{}", msg_type),
            });
        }

        if is_eq {
            let a = self.check_expression(&args[0])?;
            let b = self.check_expression(&args[1])?;
            let supported = matches!(
                a,
                Type::Int64 | Type::UInt64 | Type::Float64 | Type::Bool | Type::String
            );
            if !supported {
                return Err(TypeError::TypeMismatch {
                    expected: "Int64, UInt64, Float64, Bool or String".to_string(),
                    found: format!("{}", a),
                });
            }
            if !self.types_compatible(&a, &b) {
                return Err(TypeError::TypeMismatch {
                    expected: format!("{}", a),
                    found: format!("{}", b),
                });
            }
        } else {
            let cond = self.check_expression(&args[0])?;
            if !self.types_compatible(&Type::Bool, &cond) {
                return Err(TypeError::TypeMismatch {
                    expected: "Bool".to_string(),
                    found: format!("{}", cond),
                });
            }
        }

        Ok(Type::Void)
    }

    /// drop(box): explicit deallocation of an Option/Result box. Only the box
    /// and its cell are freed; pointer payloads stay owned by their allocation.
    /// Using the value afterwards (or dropping twice) is undefined behavior,
    /// same as std.mem `free`.
    fn check_drop_call(&mut self, args: &[Expr]) -> Result<Type, TypeError> {
        if args.len() != 1 {
            return Err(TypeError::WrongArgumentCount {
                expected: 1,
                found: args.len(),
            });
        }
        let arg_type = self.check_expression(&args[0])?;
        match arg_type {
            Type::Option(_) | Type::Result(_, _) => Ok(Type::Void),
            _ => Err(TypeError::TypeMismatch {
                expected: "Option<T> or Result<T, E>".to_string(),
                found: arg_type.to_string(),
            }),
        }
    }

    fn check_block(
        &mut self,
        stmts: &[Stmt],
        _expected_return: Option<&Type>,
    ) -> Result<Option<Type>, TypeError> {
        for (i, stmt) in stmts.iter().enumerate() {
            let is_last = i == stmts.len() - 1;
            let result = match self.check_statement(stmt) {
                Ok(r) => r,
                Err(e) => {
                    // Keep the innermost line: nested blocks would otherwise
                    // stack "line X: line Y:" wrappers.
                    if matches!(e, TypeError::AtLine { .. }) {
                        return Err(e);
                    }
                    return Err(TypeError::AtLine {
                        loc: stmt.loc(),
                        source: Box::new(e),
                    });
                }
            };

            if is_last {
                return Ok(result);
            }
        }

        Ok(None)
    }

    fn check_block_swapped(
        &mut self,
        env: TypeEnv,
        stmts: &[Stmt],
        expected_return: Option<&Type>,
    ) -> Result<Option<Type>, TypeError> {
        let prev_env = std::mem::replace(&mut self.env, env);
        let result = self.check_block(stmts, expected_return);
        self.env = prev_env;
        result
    }

    fn check_expr_swapped(&mut self, env: TypeEnv, expr: &Expr) -> Result<Type, TypeError> {
        let prev_env = std::mem::replace(&mut self.env, env);
        let result = self.check_expression(expr);
        self.env = prev_env;
        result
    }

    fn check_generic_call(
        &mut self,
        name: &str,
        info: &GenericFnInfo,
        args: &[Expr],
        expected_ret: Option<&Type>,
    ) -> Result<Type, TypeError> {
        if args.len() != info.params.len() {
            return Err(TypeError::WrongArgumentCount {
                expected: info.params.len(),
                found: args.len(),
            });
        }

        let mut arg_types = Vec::with_capacity(args.len());
        for arg in args {
            arg_types.push(self.check_expression(arg)?);
        }

        let declared_types: Vec<Type> = info.params.iter().map(|p| p.ty.clone()).collect();
        let subst = infer_with_expected(
            &info.type_params,
            &declared_types,
            &arg_types,
            info.return_type.as_ref(),
            expected_ret,
        )
        .ok_or_else(|| TypeError::CannotInferTypeArgs(name.to_string()))?;
        let key = instance_c_name(name, &info.type_params, &subst);

        if !self.instantiated.contains(&key) {
            self.instantiated.insert(key.clone());

            let concrete_params = substitute_params(&subst, &info.params);
            let mut func_env = self.env.clone();
            for p in &concrete_params {
                func_env.define_variable(p.name.clone(), p.ty.clone());
            }
            let body: Vec<Stmt> = info.body.iter().map(|s| substitute_stmt(&subst, s)).collect();
            self.check_block_swapped(func_env, &body, info.return_type.as_ref())?;
        }

        let ret = info
            .return_type
            .as_ref()
            .map(|t| substitute_type(&subst, t))
            .unwrap_or(Type::Void);
        Ok(ret)
    }

    fn check_statement(&mut self, stmt: &Stmt) -> Result<Option<Type>, TypeError> {
        match stmt {
            Stmt::Let {
                name,
                ty,
                value,
                mutable: _,
                ..
            } => {
                // A generic call with a declared type: infer remaining type
                // parameters from the annotation (e.g. `E` in `-> Result<T, E>`).
                let value_type = match (ty, value) {
                    (Some(expected), Expr::FunctionCall { name: callee, args }) => {
                        if let Expr::Identifier(n) = callee.as_ref() {
                            let fname = n.replace("::", "_");
                            if let Some(info) = self.generic_fns.get(&fname).cloned() {
                                self.check_generic_call(&fname, &info, args, Some(expected))?
                            } else {
                                self.check_expression(value)?
                            }
                        } else {
                            self.check_expression(value)?
                        }
                    }
                    _ => self.check_expression(value)?,
                };

                let stored = if let Some(expected) = ty {
                    if !self.types_compatible(expected, &value_type)
                        && !self.int_literal_satisfies(expected, &value_type, value)
                    {
                        return Err(TypeError::TypeMismatch {
                            expected: format!("{}", expected),
                            found: format!("{}", value_type),
                        });
                    }
                    // Prefer the declared type: it carries the concrete phantom
                    // parameter (e.g. Option::None() resolves to Option<Void>).
                    expected.clone()
                } else {
                    value_type
                };

                self.env.define_variable(name.clone(), stored);
                Ok(None)
            }
            Stmt::Assignment { target, value, .. } => {
                let target_type = self.check_expression(target)?;
                let value_type = self.check_expression(value)?;

                if !self.types_compatible(&target_type, &value_type)
                    && !self.int_literal_satisfies(&target_type, &value_type, value)
                {
                    return Err(TypeError::TypeMismatch {
                        expected: format!("{}", target_type),
                        found: format!("{}", value_type),
                    });
                }

                Ok(None)
            }
            Stmt::Expression(_, expr) => {
                let ty = self.check_expression(expr)?;
                Ok(Some(ty))
            }
            Stmt::Return(_, expr) => {
                if let Some(e) = expr {
                    let ty = self.check_expression(e)?;
                    Ok(Some(ty))
                } else {
                    Ok(None)
                }
            }
            Stmt::Break(_) | Stmt::Continue(_) => Ok(None),
            Stmt::Loop(_, body) => {
                self.check_block(body, None)?;
                Ok(None)
            }
            Stmt::While { condition, body, .. } => {
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
                ..
            } => {
                let iter_type = self.check_expression(iterable)?;

                // Map has no runtime backing, so iterating over it is rejected
                // here (in typechecking) rather than failing in codegen.
                let elem_type = match iter_type {
                    Type::List(elem) => *elem,
                    Type::Array(elem, _) => *elem,
                    // Iterating a Map yields its string keys.
                    Type::Map(_, _) => Type::String,
                    _ => return Err(TypeError::InvalidIterableType),
                };

                self.env.define_variable(variable.clone(), elem_type);
                self.check_block(body, None)?;
                Ok(None)
            }
            Stmt::Guard { condition, else_body, .. } => {
                let cond_type = self.check_expression(condition)?;
                if !matches!(cond_type, Type::Bool) {
                    return Err(TypeError::GuardConditionNotBool);
                }
                self.check_block(else_body, None)?;
                Ok(None)
            }
            Stmt::Spawn(_, expr) => {
                let ty = self.check_expression(expr)?;
                match ty {
                    Type::Async(_) => Ok(None),
                    _ => Err(TypeError::TypeMismatch {
                        expected: "Async<T>".to_string(),
                        found: format!("{}", ty),
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
                            _ => Err(TypeError::CannotApplyOperator(format!("{}", op))),
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
                                        expected: format!("{}", a),
                                        found: format!("{}", b),
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
                                expected: format!("{}", left_type),
                                found: format!("{}", right_type),
                            })
                        }
                    }
                    BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                        match (&left_type, &right_type) {
                            (Type::Int64, Type::Int64) => Ok(Type::Bool),
                            (Type::Float64, Type::Float64) => Ok(Type::Bool),
                            (Type::UInt64, Type::UInt64) => Ok(Type::Bool),
                            (Type::String, Type::String) => Ok(Type::Bool),
                            _ => Err(TypeError::CannotApplyOperator(format!("{}", op))),
                        }
                    }
                    BinOp::And | BinOp::Or => {
                        if matches!(left_type, Type::Bool) && matches!(right_type, Type::Bool) {
                            Ok(Type::Bool)
                        } else {
                            Err(TypeError::CannotApplyOperator(format!("{}", op)))
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
                        from: format!("{}", source_type),
                        to: format!("{}", target_type),
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

                // Polymorphic assert builtins (work on Int64/UInt64/Float64/Bool/String)
                if Self::is_polymorphic_assert(&func_name) {
                    return self.check_assert_builtin_call(&func_name, args);
                }

                // drop(box): explicit deallocation of an Option/Result box.
                if func_name == "drop" {
                    return self.check_drop_call(args);
                }

                // Generic function call: infer type arguments and check the instantiation
                if let Some(info) = self.generic_fns.get(&func_name).cloned() {
                    return self.check_generic_call(&func_name, &info, args, None);
                }

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
                    if !self.types_compatible(param_type, &arg_type)
                        && !self.int_literal_satisfies(param_type, &arg_type, arg)
                    {
                        return Err(TypeError::TypeMismatch {
                            expected: format!("{}", param_type),
                            found: format!("{}", arg_type),
                        });
                    }
                }

                let ret = sig.return_type.unwrap_or(Type::Void);
                // Calling an async function yields a lazy handle; `spawn`
                // launches it on a thread, `await` extracts the value.
                if sig.is_async {
                    Ok(Type::Async(Box::new(ret)))
                } else {
                    Ok(ret)
                }
            }

            Expr::MethodCall {
                object,
                method,
                args,
            } => {
                let object_type = self.check_expression(object)?;

                match &object_type {
                    Type::String => match method.as_str() {
                        "len" => Ok(Type::Int64),
                        "to_uint" => Ok(Type::Option(Box::new(Type::UInt64))),
                        _ => Err(TypeError::UndefinedFunction(format!("String.{}", method))),
                    },
                    Type::List(elem_type) => match method.as_str() {
                        "len" => Ok(Type::Int64),
                        "iter" => Ok(Type::Async(elem_type.clone())),
                        "free" => {
                            if args.is_empty() {
                                Ok(Type::Void)
                            } else {
                                Err(TypeError::WrongArgumentCount {
                                    expected: 0,
                                    found: args.len(),
                                })
                            }
                        }
                        "append" => {
                            if args.len() == 1 {
                                let arg_type = self.check_expression(&args[0])?;
                                if self.types_compatible(elem_type, &arg_type) {
                                    Ok(Type::Void)
                                } else {
                                    Err(TypeError::TypeMismatch {
                                        expected: format!("{}", elem_type),
                                        found: format!("{}", arg_type),
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
                    Type::Map(key_type, value_type) => match method.as_str() {
                        "len" => {
                            if args.is_empty() {
                                Ok(Type::Int64)
                            } else {
                                Err(TypeError::WrongArgumentCount {
                                    expected: 0,
                                    found: args.len(),
                                })
                            }
                        }
                        "free" => {
                            if args.is_empty() {
                                Ok(Type::Void)
                            } else {
                                Err(TypeError::WrongArgumentCount {
                                    expected: 0,
                                    found: args.len(),
                                })
                            }
                        }
                        "put" => {
                            if args.len() == 2 {
                                let kt = self.check_expression(&args[0])?;
                                if !self.types_compatible(key_type, &kt) {
                                    return Err(TypeError::TypeMismatch {
                                        expected: format!("{}", key_type),
                                        found: format!("{}", kt),
                                    });
                                }
                                let vt = self.check_expression(&args[1])?;
                                if self.types_compatible(value_type, &vt) {
                                    Ok(Type::Void)
                                } else {
                                    Err(TypeError::TypeMismatch {
                                        expected: format!("{}", value_type),
                                        found: format!("{}", vt),
                                    })
                                }
                            } else {
                                Err(TypeError::WrongArgumentCount {
                                    expected: 2,
                                    found: args.len(),
                                })
                            }
                        }
                        "get" => {
                            if args.len() == 1 {
                                let kt = self.check_expression(&args[0])?;
                                if self.types_compatible(key_type, &kt) {
                                    Ok(Type::Option(value_type.clone()))
                                } else {
                                    Err(TypeError::TypeMismatch {
                                        expected: format!("{}", key_type),
                                        found: format!("{}", kt),
                                    })
                                }
                            } else {
                                Err(TypeError::WrongArgumentCount {
                                    expected: 1,
                                    found: args.len(),
                                })
                            }
                        }
                        _ => {
                            Err(TypeError::UndefinedFunction(format!("Map.{}", method)))
                        }
                    },
                    Type::Channel(elem_type) => match method.as_str() {
                        "send" => {
                            if args.len() == 1 {
                                let arg_type = self.check_expression(&args[0])?;
                                // Stack addresses must not cross threads.
                                if matches!(arg_type, Type::Ref(_)) {
                                    return Err(TypeError::TypeMismatch {
                                        expected: "owned value".to_string(),
                                        found: arg_type.to_string(),
                                    });
                                }
                                if self.types_compatible(elem_type, &arg_type) {
                                    // true = queued, false = channel closed
                                    Ok(Type::Bool)
                                } else {
                                    Err(TypeError::TypeMismatch {
                                        expected: format!("{}", elem_type),
                                        found: format!("{}", arg_type),
                                    })
                                }
                            } else {
                                Err(TypeError::WrongArgumentCount {
                                    expected: 1,
                                    found: args.len(),
                                })
                            }
                        }
                        "recv" => {
                            if args.is_empty() {
                                Ok(Type::Option(elem_type.clone()))
                            } else {
                                Err(TypeError::WrongArgumentCount {
                                    expected: 0,
                                    found: args.len(),
                                })
                            }
                        }
                        "close" => {
                            if args.is_empty() {
                                Ok(Type::Void)
                            } else {
                                Err(TypeError::WrongArgumentCount {
                                    expected: 0,
                                    found: args.len(),
                                })
                            }
                        }
                        _ => Err(TypeError::UndefinedFunction(format!("Channel.{}", method))),
                    },
                    Type::Custom(type_name) => {
                        // Check if method exists in impl blocks.
                        // Method signature is receiver-first: the first param
                        // must match the object type, explicit args follow.
                        let full_name = format!("{}.{}", type_name, method);
                        if let Some(sig) = self.env.get_function(&full_name).cloned() {
                            if sig.params.is_empty() {
                                return Err(TypeError::UndefinedFunction(format!(
                                    "{}.{}",
                                    type_name, method
                                )));
                            }
                            let receiver_ty = &sig.params[0].1;
                            let object_check = match &object_type {
                                Type::Ref(inner) => inner.as_ref(),
                                other => other,
                            };
                            let receiver_check = match receiver_ty {
                                Type::Ref(inner) => inner.as_ref(),
                                other => other,
                            };
                            if !self.types_compatible(receiver_check, object_check) {
                                return Err(TypeError::TypeMismatch {
                                    expected: format!("{}", receiver_ty),
                                    found: format!("{}", object_type),
                                });
                            }
                            if args.len() != sig.params.len() - 1 {
                                return Err(TypeError::WrongArgumentCount {
                                    expected: sig.params.len() - 1,
                                    found: args.len(),
                                });
                            }
                            for (arg, (_, param_type, _)) in args.iter().zip(&sig.params[1..]) {
                                let arg_type = self.check_expression(arg)?;
                                if !self.types_compatible(param_type, &arg_type)
                                    && !self.int_literal_satisfies(param_type, &arg_type, arg)
                                {
                                    return Err(TypeError::TypeMismatch {
                                        expected: format!("{}", param_type),
                                        found: format!("{}", arg_type),
                                    });
                                }
                            }
                            let ret = sig.return_type.unwrap_or(Type::Void);
                            if sig.is_async {
                                Ok(Type::Async(Box::new(ret)))
                            } else {
                                Ok(ret)
                            }
                        } else {
                            Err(TypeError::UndefinedFunction(format!("{}.{}", type_name, method)))
                        }
                    }
                    _ => Err(TypeError::CannotCallNonFunction(format!("{}", object_type))),
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
                        // Slice: xs[a..b] / xs[a..=b] -> List<T>
                        if matches!(&**index, Expr::Range { .. }) {
                            match &index_type {
                                Type::List(range_elem) => match range_elem.as_ref() {
                                    Type::Int64 | Type::UInt64 => {
                                        Ok(Type::List(elem_type.clone()))
                                    }
                                    other => Err(TypeError::TypeMismatch {
                                        expected: "Int64 or UInt64 range".to_string(),
                                        found: format!("{}", other),
                                    }),
                                },
                                other => Err(TypeError::TypeMismatch {
                                    expected: "range".to_string(),
                                    found: format!("{}", other),
                                }),
                            }
                        } else if matches!(index_type, Type::UInt64 | Type::Int64) {
                            Ok((**elem_type).clone())
                        } else {
                            Err(TypeError::TypeMismatch {
                                expected: "UInt64 or Int64".to_string(),
                                found: format!("{}", index_type),
                            })
                        }
                    }
                    Type::Map(key_type, value_type) => {
                        if self.types_compatible(&key_type, &index_type) {
                            Ok((**value_type).clone())
                        } else {
                            Err(TypeError::TypeMismatch {
                                expected: format!("{}", key_type),
                                found: format!("{}", index_type),
                            })
                        }
                    }
                    Type::Array(elem_type, _) => {
                        if matches!(index_type, Type::UInt64 | Type::Int64) {
                            Ok((**elem_type).clone())
                        } else {
                            Err(TypeError::TypeMismatch {
                                expected: "UInt64 or Int64".to_string(),
                                found: format!("{}", index_type),
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
                        expected: format!("{}", start_type),
                        found: format!("{}", end_type),
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
                            expected: format!("{}", first_type),
                            found: format!("{}", elem_type),
                        });
                    }
                }
                Ok(Type::List(Box::new(first_type)))
            }

            Expr::MapLiteral(pairs) => {
                if pairs.is_empty() {
                    return Ok(Type::Map(
                        Box::new(Type::String),
                        Box::new(Type::Void),
                    ));
                }
                let (first_key, first_value) = &pairs[0];
                let key_type = self.check_expression(first_key)?;
                if !matches!(key_type, Type::String) && !matches!(key_type, Type::Ref(_)) {
                    return Err(TypeError::TypeMismatch {
                        expected: "String".to_string(),
                        found: format!("{}", key_type),
                    });
                }
                let value_type = self.check_expression(first_value)?;
                for (k, v) in &pairs[1..] {
                    let kt = self.check_expression(k)?;
                    if !self.types_compatible(&key_type, &kt) {
                        return Err(TypeError::TypeMismatch {
                            expected: format!("{}", key_type),
                            found: format!("{}", kt),
                        });
                    }
                    let vt = self.check_expression(v)?;
                    if !self.types_compatible(&value_type, &vt) {
                        return Err(TypeError::MappingValueMismatch {
                            expected: format!("{}", value_type),
                            found: format!("{}", vt),
                        });
                    }
                }
                Ok(Type::Map(Box::new(Type::String), Box::new(value_type)))
            }

            Expr::Match { expr, arms } => {
                let expr_type = self.check_expression(expr)?;

                // Unwrap reference type for pattern matching
                let match_type = match &expr_type {
                    Type::Ref(inner) => (**inner).clone(),
                    _ => expr_type.clone(),
                };

                let mut result_ty: Option<Type> = None;
                for arm in arms {
                    let mut arm_env = self.env.clone();
                    self.check_pattern(&arm.pattern, &match_type, &mut arm_env)?;

                    // Check guard if present
                    if let Some(guard) = &arm.guard {
                        let guard_type = self.check_expr_swapped(arm_env.clone(), guard)?;
                        if !matches!(guard_type, Type::Bool) {
                            return Err(TypeError::GuardConditionNotBool);
                        }
                    }

                    let body_ty = self.check_expr_swapped(arm_env, &arm.body)?;
                    if result_ty.is_none() {
                        result_ty = Some(body_ty);
                    } else if let Some(prev) = &result_ty {
                        // A Void arm body (e.g. an empty-branch block) must not
                        // downgrade the common type of the match expression.
                        if !matches!(prev, Type::Void) && !matches!(body_ty, Type::Void) {
                            if !self.types_compatible(prev, &body_ty) {
                                return Err(TypeError::TypeMismatch {
                                    expected: format!("{}", prev),
                                    found: format!("{}", body_ty),
                                });
                            }
                        }
                    }
                }

                Ok(result_ty.unwrap_or(Type::Void))
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
                        found: format!("{}", cond_type),
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
                // New scope for the whole block: clone the outer env, run all
                // statements sequentially in it, then restore the outer env.
                // Generic state lives on `self` and is preserved throughout.
                let outer_env = self.env.clone();
                self.env = outer_env.clone();
                let mut outcome: Result<Type, TypeError> = Ok(Type::Void);
                for (i, stmt) in stmts.iter().enumerate() {
                    match self.check_statement(stmt) {
                        Ok(t) => {
                            if i == stmts.len() - 1 {
                                outcome = Ok(t.unwrap_or(Type::Void));
                            }
                        }
                        Err(e) => {
                            outcome = Err(e);
                            break;
                        }
                    }
                }
                self.env = outer_env;
                outcome
            }

            Expr::ChannelBounded { elem_type, capacity } => {
                let cap_type = self.check_expression(capacity)?;
                if !matches!(cap_type, Type::UInt64 | Type::Int64) {
                    return Err(TypeError::TypeMismatch {
                        expected: "UInt64 or Int64".to_string(),
                        found: format!("{}", cap_type),
                    });
                }
                Ok(Type::Channel(elem_type.clone()))
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
                                    expected: format!("{}", expected_type),
                                    found: format!("{}", field_type),
                                });
                            }
                        }
                        Ok(Type::Custom(name.clone()))
                    }
                    _ => Err(TypeError::TypeMismatch {
                        expected: "Struct type".to_string(),
                        found: format!("{}", name),
                    }),
                }
            }

            Expr::EnumInit { enum_name, variant, args } => {
                // Built-in phantom enums: Option::Some/None, Result::Ok/Err
                if enum_name == "Option" {
                    return match variant.as_str() {
                        "Some" => {
                            if args.len() != 1 {
                                return Err(TypeError::WrongArgumentCount {
                                    expected: 1,
                                    found: args.len(),
                                });
                            }
                            let arg_type = self.check_expression(&args[0])?;
                            Ok(Type::Option(Box::new(arg_type)))
                        }
                        "None" => {
                            if !args.is_empty() {
                                return Err(TypeError::WrongArgumentCount {
                                    expected: 0,
                                    found: args.len(),
                                });
                            }
                            Ok(Type::Option(Box::new(Type::Void)))
                        }
                        _ => Err(TypeError::UndefinedVariable(format!(
                            "Option::{}",
                            variant
                        ))),
                    };
                }
                if enum_name == "Result" {
                    return match variant.as_str() {
                        "Ok" => {
                            if args.len() != 1 {
                                return Err(TypeError::WrongArgumentCount {
                                    expected: 1,
                                    found: args.len(),
                                });
                            }
                            let arg_type = self.check_expression(&args[0])?;
                            Ok(Type::Result(Box::new(arg_type), Box::new(Type::Void)))
                        }
                        "Err" => {
                            if args.len() != 1 {
                                return Err(TypeError::WrongArgumentCount {
                                    expected: 1,
                                    found: args.len(),
                                });
                            }
                            let arg_type = self.check_expression(&args[0])?;
                            Ok(Type::Result(Box::new(Type::Void), Box::new(arg_type)))
                        }
                        _ => Err(TypeError::UndefinedVariable(format!(
                            "Result::{}",
                            variant
                        ))),
                    };
                }

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
                                if !self.types_compatible(expected_type, &arg_type)
                                    && !self.int_literal_satisfies(expected_type, &arg_type, arg)
                                {
                                    return Err(TypeError::TypeMismatch {
                                        expected: format!("{}", expected_type),
                                        found: format!("{}", arg_type),
                                    });
                                }
                            }
                        }

                        Ok(Type::Custom(enum_name.clone()))
                    }
                    _ => Err(TypeError::TypeMismatch {
                        expected: "Enum type".to_string(),
                        found: format!("{}", enum_name),
                    }),
                }
            }

            Expr::Await(expr) => {
                let expr_type = self.check_expression(expr)?;
                match expr_type {
                    Type::Async(inner) => Ok(*inner),
                    _ => Err(TypeError::TypeMismatch {
                        expected: "Async<T>".to_string(),
                        found: format!("{}", expr_type),
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
                        found: format!("{}", expected_type),
                    })
                }
            }
            Pattern::StringLiteral(_) => {
                if matches!(expected_type, Type::String) {
                    Ok(())
                } else {
                    Err(TypeError::TypeMismatch {
                        expected: "String".to_string(),
                        found: format!("{}", expected_type),
                    })
                }
            }
            Pattern::BoolLiteral(_) => {
                if matches!(expected_type, Type::Bool) {
                    Ok(())
                } else {
                    Err(TypeError::TypeMismatch {
                        expected: "Bool".to_string(),
                        found: format!("{}", expected_type),
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
                    Type::Option(inner) => {
                        match variant.as_str() {
                            "Some" => {
                                if data.as_ref().map_or(false, |d| d.len() != 1)
                                    || data.is_none()
                                {
                                    return Err(TypeError::WrongArgumentCount {
                                        expected: 1,
                                        found: data.as_ref().map_or(0, |d| d.len()),
                                    });
                                }
                                self.check_pattern(&data.as_ref().unwrap()[0], inner, env)?;
                                Ok(())
                            }
                            "None" => {
                                if data.is_some() {
                                    return Err(TypeError::WrongArgumentCount {
                                        expected: 0,
                                        found: 1,
                                    });
                                }
                                Ok(())
                            }
                            _ => Err(TypeError::UndefinedVariable(format!(
                                "Option::{}",
                                variant
                            ))),
                        }
                    }
                    Type::Result(ok_ty, err_ty) => {
                        match variant.as_str() {
                            "Ok" => {
                                if data.as_ref().map_or(false, |d| d.len() != 1)
                                    || data.is_none()
                                {
                                    return Err(TypeError::WrongArgumentCount {
                                        expected: 1,
                                        found: data.as_ref().map_or(0, |d| d.len()),
                                    });
                                }
                                self.check_pattern(&data.as_ref().unwrap()[0], ok_ty, env)?;
                                Ok(())
                            }
                            "Err" => {
                                if data.as_ref().map_or(false, |d| d.len() != 1)
                                    || data.is_none()
                                {
                                    return Err(TypeError::WrongArgumentCount {
                                        expected: 1,
                                        found: data.as_ref().map_or(0, |d| d.len()),
                                    });
                                }
                                self.check_pattern(&data.as_ref().unwrap()[0], err_ty, env)?;
                                Ok(())
                            }
                            _ => Err(TypeError::UndefinedVariable(format!(
                                "Result::{}",
                                variant
                            ))),
                        }
                    }
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
                        found: format!("{}", expected_type),
                    }),
                }
            }
            Pattern::Wildcard => Ok(()),
        }
    }

    /// A plain integer literal (typed `Int64` by default) satisfies a declared
    /// `UInt64`/`Float64` expectation, matching how `5` works for any Rust int.
    fn int_literal_satisfies(&self, expected: &Type, found: &Type, expr: &Expr) -> bool {
        matches!(expr, Expr::IntegerLiteral(_))
            && matches!(found, Type::Int64)
            && matches!(expected, Type::Int64 | Type::UInt64 | Type::Float64)
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
            (Type::List(a), Type::List(b)) => {
                self.is_unknown(a) || self.is_unknown(b) || self.types_compatible(a, b)
            }
            (Type::Map(k1, v1), Type::Map(k2, v2)) => {
                self.types_compatible(k1, k2) && self.types_compatible(v1, v2)
            }
            (Type::Result(o1, e1), Type::Result(o2, e2)) => {
                (self.is_unknown(e1) || self.is_unknown(e2) || self.types_compatible(e1, e2))
                    && (self.is_unknown(o1) || self.is_unknown(o2) || self.types_compatible(o1, o2))
            }
            (Type::Option(a), Type::Option(b)) => {
                self.is_unknown(a) || self.is_unknown(b) || self.types_compatible(a, b)
            }
            (Type::Async(a), Type::Async(b)) => self.types_compatible(a, b),
            (Type::Channel(a), Type::Channel(b)) => self.types_compatible(a, b),
            (Type::Ref(a), Type::Ref(b)) => self.types_compatible(a, b),
            (Type::Custom(a), Type::Custom(b)) => a == b,
            (Type::Array(a, s1), Type::Array(b, s2)) => self.types_compatible(a, b) && s1 == s2,
            (Type::Generic(a), Type::Generic(b)) => a == b,
            _ => false,
        }
    }

    /// A `Void` type inside Option/Result acts as an unknown placeholder:
    /// `Option::None()` is `Option<Void>` and unifies with `Option<Int64>`.
    fn is_unknown(&self, t: &Type) -> bool {
        matches!(t, Type::Void)
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

    #[test]
    fn test_type_check_impl_method() {
        let input = "\
@struct Point { x: Float64, y: Float64 }
@impl Point {
    @fn norm(p: Point) -> Float64 { return sqrt(p.x * p.x + p.y * p.y); }
    @fn scaled(p: Point, k: Float64) -> Point { return Point { x: p.x * k, y: p.y * k }; }
}
@fn main() -> Void {
    let p: Point = Point { x: 3.0, y: 4.0 };
    let n: Float64 = p.norm();
    let q: Point = p.scaled(2.0);
    print_float(q.y);
    print_float(n);
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_type_check_list_index() {
        let input = "\
@struct P { x: Float64, y: Float64 }
@fn main() -> Void {
    let arr: List<Int64> = [1, 2, 3];
    let n: Int64 = arr[0];
    let points: List<P> = [P { x: 1.0, y: 2.0 }];
    let py: Float64 = points[0].y;
    print_float(py);
    print_int(n);
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_type_check_result_option() {
        let input = "\
@fn divide(a: Int64, b: Int64) -> Result<Int64, String> {
    if b == 0 {
        return Result::Err(\"div0\");
    }
    return Result::Ok(a / b);
}
@fn main() -> Void {
    let r: Result<Int64, String> = divide(10, 2);
    let q: Int64 = match r {
        | Ok(v) => v,
        | Err(e) => -1
    };
    let absent: Option<Int64> = Option::None;
    let z: Int64 = match absent {
        | None => 42,
        | Some(v) => v
    };
    let present: Option<Int64> = Option::Some(7);
    let w: Int64 = match present {
        | Some(v) => v,
        | None => 0
    };
    let p: Option<Float64> = parse_float(\"1.5\");
    let f: Float64 = match p {
        | Some(v) => v,
        | None => 0.0
    };
    print_int(q);
    print_int(z);
    print_int(w);
    print_float(f);
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_type_check_result_option_bad_variant() {
        let input = "\
@fn main() -> Void {
    let r: Result<Int64, String> = Result::Ok(1);
    let q: Int64 = match r {
        | Some(v) => v,
        | None => 0
    };
    print_int(q);
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_err());
    }

    #[test]
    fn test_type_check_generic_identity() {
        let input = "\
@fn identity<T>(x: T) -> T {
    return x;
}
@fn main() -> Void {
    let a: Int64 = identity(5);
    let b: Float64 = identity(2.5);
    let c: String = identity(\"hi\");
    print_int(a);
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_type_check_generic_nested_and_chained() {
        let input = "\
@fn first<T, K>(a: T, b: K) -> T {
    return a;
}
@fn wrap<T>(x: T) -> Option<T> {
    return Option::Some(x);
}
@fn double_it<T>(x: T) -> T {
    return first(x, x);
}
@fn main() -> Void {
    let a: Int64 = double_it(21);
    let m: Option<Int64> = wrap(a);
    let s: String = first(\"hello\", 42);
    print_int(a);
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_type_check_generic_return_only_param_from_let() {
        let input = "\
@fn ok_wrap<T, E>(x: T) -> Result<T, E> {
    return Result::Ok(x);
}
@fn main() -> Void {
    let r: Result<String, Int64> = ok_wrap(\"fine\");
    print(\"ok\");
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_type_check_generic_conflict() {
        let input = "\
@fn same<T>(a: T, b: T) -> T {
    return a;
}
@fn main() -> Void {
    let x: Int64 = same(1, \"two\");
    print_int(x);
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_err());
    }

    #[test]
    fn test_type_check_generic_ambiguous() {
        // `T` appears neither in parameters nor in any annotation: no way to infer.
        let input = "\
@fn make<T>() -> T {
    return 0;
}
@fn main() -> Void {
    let x = make();
    print_int(x);
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_err());
    }

    #[test]
    fn test_type_check_async_spawn_await() {
        let input = "\
@fn async fetch(url: String) -> String {
    return url;
}
@fn async add_async(a: Int64, b: Int64) -> Int64 {
    return a + b;
}
@fn main() -> Void {
    let h: Async<String> = fetch(\"hi\");
    spawn h;
    let s: String = h await;
    let sum: Int64 = add_async(20, 22) await;
    println(s);
    print_int(sum);
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_type_check_channel_send_recv_close() {
        let input = "\
@fn main() -> Void {
    let ch: Channel<Int64> = Channel<Int64>(4);
    let ok: Bool = ch.send(1);
    let m: Option<Int64> = ch.recv();
    ch.close();
    let v: Int64 = match m {
        | Some(x) => x,
        | None => -1
    };
    print_int(v);
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_type_check_spawn_non_async() {
        let input = "\
@fn main() -> Void {
    spawn 42;
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_err());
    }

    #[test]
    fn test_type_check_await_non_async() {
        let input = "\
@fn main() -> Void {
    let x: Int64 = 5;
    let y: Int64 = x await;
    print_int(y);
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_err());
    }

    #[test]
    fn test_type_check_channel_send_wrong_type() {
        let input = "\
@fn main() -> Void {
    let ch: Channel<Int64> = Channel<Int64>(4);
    ch.send(\"oops\");
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_err());
    }

    #[test]
    fn test_type_check_async_generic_rejected() {
        let input = "\
@fn async wrap<T>(x: T) -> T {
    return x;
}
@fn main() -> Void {
    let a: Int64 = wrap(1) await;
    print_int(a);
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_err());
    }

        #[test]
    fn test_type_check_list_slice() {
        let input = "\
@fn main() -> Void {
    let xs: List<Int64> = [10, 20, 30, 40];
    let a: List<Int64> = xs[1..3];
    let b: List<Int64> = xs[0..=1];
    let n: Int64 = a.len();
    print_int(n);
    print_int(b[0]);
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_type_check_for_map_yields_keys() {
        // Iterating over a Map yields its string keys.
        let input = "\
@fn f(m: Map<String, Int64>) -> Void {
    for k in m {
        print(k);
    }
}
@fn main() -> Void {}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_type_check_map_literal_and_methods() {
        let input = "\
@fn main() -> Void {
    let m: Map<String, Int64> = #{ \"alice\": 90, \"bob\": 75 };
    let n: Int64 = m.len();
    m.put(\"carol\", 60);
    let got: Option<Int64> = m.get(\"dave\");
    m[\"carol\"] = 61;
    drop(got);
    m.free();
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        let result = checker.check_program(&program);
        assert!(result.is_ok(), "{}", format!("{:?}", result));
    }

    #[test]
    fn test_type_check_map_value_mismatch_rejected() {
        let input = "\
@fn main() -> Void {
    let m: Map<String, Int64> = #{ \"a\": 1, \"b\": \"no\" };
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        let result = checker.check_program(&program);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("Map value type mismatch"));
    }

    #[test]
    fn test_type_check_list_dynamic_ops() {
        let input = "\
@fn sum_all(xs: List<Int64>) -> Int64 {
    let total: Int64 = 0;
    for x in xs {
        total = total + x;
    }
    return total;
}
@fn main() -> Void {
    let dyn: List<Int64> = [1, 2, 3];
    let n: Int64 = dyn.len();
    dyn.append(4);
    let doubled: List<Int64> = dyn ++ dyn;
    let r: List<Int64> = 0..10;
    print_int(n);
    print_int(sum_all(doubled));
    print_int(r.len());
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_type_check_drop() {
        let input = "\
@fn main() -> Void {
    let m: Option<Int64> = parse_int(\"42\");
    drop(m);
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_ok());
    }

    #[test]
    fn test_type_check_drop_non_box() {
        let input = "\
@fn main() -> Void {
    drop(42);
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_err());
    }

    #[test]
    fn test_type_check_channel_send_ref_rejected() {
        let input = "\
@fn main() -> Void {
    let ch: Channel<Int64> = Channel<Int64>(4);
    let x: Int64 = 5;
    ch.send(&x);
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_err());
    }

    #[test]
    fn test_type_check_impl_method_bad_args() {        let input = "\
@struct Point { x: Float64, y: Float64 }
@impl Point {
    @fn norm(p: Point) -> Float64 { return p.x; }
}
@fn main() -> Void {
    let p: Point = Point { x: 3.0, y: 4.0 };
    let n: Float64 = p.norm(1.0);
    print_float(n);
}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut checker = TypeChecker::new();
        assert!(checker.check_program(&program).is_err());
    }
}
