use crate::ast::*;
use std::collections::HashMap;

/// A generic function as declared in source. Type parameters are matched by name
/// and substituted with concrete types during monomorphization.
#[derive(Debug, Clone)]
pub struct GenericFnInfo {
    pub type_params: Vec<String>,
    pub params: Vec<FunctionParam>,
    pub return_type: Option<Type>,
    pub body: Vec<Stmt>,
}

/// Does the type contain a `Type::Generic` anywhere (directly or nested)?
pub fn contains_generic(ty: &Type) -> bool {
    match ty {
        Type::Generic(_) => true,
        Type::List(inner) => contains_generic(inner),
        Type::Map(k, v) => contains_generic(k) || contains_generic(v),
        Type::Result(ok, err) => contains_generic(ok) || contains_generic(err),
        Type::Option(inner) => contains_generic(inner),
        Type::Async(inner) => contains_generic(inner),
        Type::Channel(inner) => contains_generic(inner),
        Type::Array(inner, _) => contains_generic(inner),
        Type::Ref(inner) => contains_generic(inner),
        _ => false,
    }
}

/// Substitute type parameters in a type. Unknown parameters are left as-is
/// (only in-scope parameters may exist after successful inference).
pub fn substitute_type(subst: &HashMap<String, Type>, ty: &Type) -> Type {
    match ty {
        Type::Generic(name) => subst.get(name).cloned().unwrap_or_else(|| ty.clone()),
        Type::List(inner) => Type::List(Box::new(substitute_type(subst, inner))),
        Type::Map(k, v) => Type::Map(
            Box::new(substitute_type(subst, k)),
            Box::new(substitute_type(subst, v)),
        ),
        Type::Result(ok, err) => Type::Result(
            Box::new(substitute_type(subst, ok)),
            Box::new(substitute_type(subst, err)),
        ),
        Type::Option(inner) => Type::Option(Box::new(substitute_type(subst, inner))),
        Type::Async(inner) => Type::Async(Box::new(substitute_type(subst, inner))),
        Type::Channel(inner) => Type::Channel(Box::new(substitute_type(subst, inner))),
        Type::Array(inner, size) => Type::Array(Box::new(substitute_type(subst, inner)), *size),
        Type::Ref(inner) => Type::Ref(Box::new(substitute_type(subst, inner))),
        other => other.clone(),
    }
}

pub fn substitute_params(subst: &HashMap<String, Type>, params: &[FunctionParam]) -> Vec<FunctionParam> {
    params
        .iter()
        .map(|p| FunctionParam {
            name: p.name.clone(),
            ty: substitute_type(subst, &p.ty),
            is_move: p.is_move,
        })
        .collect()
}

pub fn substitute_stmt(subst: &HashMap<String, Type>, stmt: &Stmt) -> Stmt {
    match stmt {
        Stmt::Let {
            loc,
            name,
            ty,
            value,
            mutable,
        } => Stmt::Let {
            loc: *loc,
            name: name.clone(),
            ty: ty.as_ref().map(|t| substitute_type(subst, t)),
            value: substitute_expr(subst, value),
            mutable: *mutable,
        },
        Stmt::Assignment {
            loc,
            target,
            value,
        } => Stmt::Assignment {
            loc: *loc,
            target: substitute_expr(subst, target),
            value: substitute_expr(subst, value),
        },
        Stmt::Expression(loc, e) => Stmt::Expression(*loc, substitute_expr(subst, e)),
        Stmt::Return(loc, Some(e)) => Stmt::Return(*loc, Some(substitute_expr(subst, e))),
        Stmt::Return(loc, None) => Stmt::Return(*loc, None),
        Stmt::Break(loc) => Stmt::Break(*loc),
        Stmt::Continue(loc) => Stmt::Continue(*loc),
        Stmt::Loop(loc, body) => {
            Stmt::Loop(*loc, body.iter().map(|s| substitute_stmt(subst, s)).collect())
        }
        Stmt::While {
            loc,
            condition,
            body,
        } => Stmt::While {
            loc: *loc,
            condition: substitute_expr(subst, condition),
            body: body.iter().map(|s| substitute_stmt(subst, s)).collect(),
        },
        Stmt::For {
            loc,
            variable,
            iterable,
            body,
        } => Stmt::For {
            loc: *loc,
            variable: variable.clone(),
            iterable: substitute_expr(subst, iterable),
            body: body.iter().map(|s| substitute_stmt(subst, s)).collect(),
        },
        Stmt::Guard {
            loc,
            condition,
            else_body,
        } => Stmt::Guard {
            loc: *loc,
            condition: substitute_expr(subst, condition),
            else_body: else_body.iter().map(|s| substitute_stmt(subst, s)).collect(),
        },
        Stmt::Spawn(loc, e) => Stmt::Spawn(*loc, substitute_expr(subst, e)),
        Stmt::Select { loc, arms } => Stmt::Select {
            loc: *loc,
            arms: arms
                .iter()
                .map(|arm| SelectArm {
                    event: match &arm.event {
                        SelectEvent::Recv { target, name, ty } => SelectEvent::Recv {
                            target: substitute_expr(subst, target),
                            name: name.clone(),
                            ty: substitute_type(subst, ty),
                        },
                        SelectEvent::Await { target, name, ty } => SelectEvent::Await {
                            target: substitute_expr(subst, target),
                            name: name.clone(),
                            ty: substitute_type(subst, ty),
                        },
                        SelectEvent::Timeout(ms) => {
                            SelectEvent::Timeout(substitute_expr(subst, ms))
                        }
                        SelectEvent::Default => SelectEvent::Default,
                    },
                    body: arm.body.iter().map(|s| substitute_stmt(subst, s)).collect(),
                })
                .collect(),
        },
    }
}

pub fn substitute_expr(subst: &HashMap<String, Type>, expr: &Expr) -> Expr {
    match expr {
        Expr::IntegerLiteral(v, sp) => Expr::IntegerLiteral(*v, *sp),
        Expr::FloatLiteral(v, sp) => Expr::FloatLiteral(*v, *sp),
        Expr::StringLiteral(s, sp) => Expr::StringLiteral(s.clone(), *sp),
        Expr::BoolLiteral(b, sp) => Expr::BoolLiteral(*b, *sp),
        Expr::Identifier(n, sp) => Expr::Identifier(n.clone(), *sp),
        Expr::BinaryOp { op, left, right, .. } => Expr::BinaryOp {
            span: expr.span(),
            op: op.clone(),
            left: Box::new(substitute_expr(subst, left)),
            right: Box::new(substitute_expr(subst, right)),
        },
        Expr::UnaryOp { op, expr: e, .. } => Expr::UnaryOp {
            span: expr.span(),
            op: op.clone(),
            expr: Box::new(substitute_expr(subst, e)),
        },
        Expr::Ref(e, sp) => Expr::Ref(Box::new(substitute_expr(subst, e)), *sp),
        Expr::Cast { expr: e, target_type, .. } => Expr::Cast {
            span: expr.span(),
            expr: Box::new(substitute_expr(subst, e)),
            target_type: substitute_type(subst, target_type),
        },
        Expr::FunctionCall { name, args, .. } => Expr::FunctionCall {
            span: expr.span(),
            name: Box::new(substitute_expr(subst, name)),
            args: args.iter().map(|a| substitute_expr(subst, a)).collect(),
        },
        Expr::MethodCall {
            object,
            method,
            args,
            ..
        } => Expr::MethodCall {
            span: expr.span(),
            object: Box::new(substitute_expr(subst, object)),
            method: method.clone(),
            args: args.iter().map(|a| substitute_expr(subst, a)).collect(),
        },
        Expr::FieldAccess { object, field, .. } => Expr::FieldAccess {
            span: expr.span(),
            object: Box::new(substitute_expr(subst, object)),
            field: field.clone(),
        },
        Expr::IndexAccess { object, index, .. } => Expr::IndexAccess {
            span: expr.span(),
            object: Box::new(substitute_expr(subst, object)),
            index: Box::new(substitute_expr(subst, index)),
        },
        Expr::StructInit { name, fields, .. } => Expr::StructInit {
            span: expr.span(),
            name: name.clone(),
            fields: fields
                .iter()
                .map(|(f, e)| (f.clone(), substitute_expr(subst, e)))
                .collect(),
        },
        Expr::EnumInit {
            enum_name,
            variant,
            args,
            ..
        } => Expr::EnumInit {
            span: expr.span(),
            enum_name: enum_name.clone(),
            variant: variant.clone(),
            args: args.iter().map(|a| substitute_expr(subst, a)).collect(),
        },
        Expr::Match { expr: e, arms, .. } => Expr::Match {
            span: expr.span(),
            expr: Box::new(substitute_expr(subst, e)),
            arms: arms
                .iter()
                .map(|arm| MatchArm {
                    pattern: arm.pattern.clone(),
                    guard: arm.guard.as_ref().map(|g| substitute_expr(subst, g)),
                    body: substitute_expr(subst, &arm.body),
                })
                .collect(),
        },
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => Expr::If {
            span: expr.span(),
            condition: Box::new(substitute_expr(subst, condition)),
            then_branch: Box::new(substitute_expr(subst, then_branch)),
            else_branch: else_branch
                .as_ref()
                .map(|e| Box::new(substitute_expr(subst, e))),
        },
        Expr::Block(stmts, sp) => Expr::Block(stmts.iter().map(|s| substitute_stmt(subst, s)).collect(), *sp),
        Expr::Range {
            start,
            end,
            inclusive,
            ..
        } => Expr::Range {
            span: expr.span(),
            start: Box::new(substitute_expr(subst, start)),
            end: Box::new(substitute_expr(subst, end)),
            inclusive: *inclusive,
        },
        Expr::ArrayLiteral(elems, sp) => Expr::ArrayLiteral(
            elems.iter().map(|e| substitute_expr(subst, e)).collect(), *sp,
        ),
        Expr::MapLiteral(pairs, sp) => Expr::MapLiteral(
            pairs
                .iter()
                .map(|(k, v)| {
                    (
                        substitute_expr(subst, k),
                        substitute_expr(subst, v),
                    )
                })
                .collect(), *sp,
        ),
        Expr::ChannelBounded { elem_type, capacity, .. } => Expr::ChannelBounded {
            span: expr.span(),
            elem_type: Box::new(substitute_type(subst, elem_type)),
            capacity: Box::new(substitute_expr(subst, capacity)),
        },
        Expr::Await(e, sp) => Expr::Await(Box::new(substitute_expr(subst, e)), *sp),
    }
}

/// Unify a declared type (possibly containing generics) against a concrete
/// argument type, binding generics into `subst`. Returns false on mismatch or
/// when the argument is not fully concrete.
fn unify(subst: &mut HashMap<String, Type>, declared: &Type, arg: &Type) -> bool {
    match declared {
        Type::Generic(p) => {
            if contains_generic(arg) {
                return false;
            }
            // A bare Void is an "unknown" placeholder (empty list, None),
            // never a real inference result.
            if matches!(arg, Type::Void) {
                return false;
            }
            match subst.get(p) {
                Some(bound) => bound == arg,
                None => {
                    subst.insert(p.clone(), arg.clone());
                    true
                }
            }
        }
        Type::List(d) => match arg {
            Type::List(a) => unify(subst, d, a),
            _ => false,
        },
        Type::Map(dk, dv) => match arg {
            Type::Map(ak, av) => unify(subst, dk, ak) && unify(subst, dv, av),
            _ => false,
        },
        Type::Result(dok, derr) => match arg {
            Type::Result(aok, aerr) => unify(subst, dok, aok) && unify(subst, derr, aerr),
            _ => false,
        },
        Type::Option(d) => match arg {
            Type::Option(a) => unify(subst, d, a),
            _ => false,
        },
        Type::Async(d) => match arg {
            Type::Async(a) => unify(subst, d, a),
            _ => false,
        },
        Type::Channel(d) => match arg {
            Type::Channel(a) => unify(subst, d, a),
            _ => false,
        },
        Type::Array(d, size) => match arg {
            Type::Array(a, asize) => size == asize && unify(subst, d, a),
            _ => false,
        },
        Type::Ref(d) => match arg {
            Type::Ref(a) => unify(subst, d, a),
            _ => false,
        },
        other => other == arg,
    }
}

/// Infer a substitution that maps each type parameter to the concrete type of the
/// corresponding argument. Returns `None` if inference fails or is ambiguous
/// (a type parameter that never appears in a parameter type cannot be inferred).
pub fn infer_substitution(
    type_params: &[String],
    declared_param_types: &[Type],
    arg_types: &[Type],
) -> Option<HashMap<String, Type>> {
    infer_with_expected(type_params, declared_param_types, arg_types, None, None)
}

/// Same as [`infer_substitution`], but additionally unifies the declared return
/// type against an expected type (e.g. a `let` annotation). This allows type
/// parameters that only occur in the return type to be inferred from context.
pub fn infer_with_expected(
    type_params: &[String],
    declared_param_types: &[Type],
    arg_types: &[Type],
    declared_ret: Option<&Type>,
    expected_ret: Option<&Type>,
) -> Option<HashMap<String, Type>> {
    let mut subst: HashMap<String, Type> = HashMap::new();
    for (declared, arg) in declared_param_types.iter().zip(arg_types.iter()) {
        if !unify(&mut subst, declared, arg) {
            return None;
        }
    }
    if let (Some(decl), Some(exp)) = (declared_ret, expected_ret) {
        if !unify(&mut subst, decl, exp) {
            return None;
        }
    }
    for p in type_params {
        if !subst.contains_key(p) {
            return None;
        }
    }
    Some(subst)
}

/// Short, readable mangling of a concrete type for instantiated function names.
pub fn mangle_type(ty: &Type) -> String {
    match ty {
        Type::Int64 => "i64".to_string(),
        Type::UInt64 => "u64".to_string(),
        Type::Float64 => "f64".to_string(),
        Type::Bool => "bool".to_string(),
        Type::String => "str".to_string(),
        Type::Bytes => "bytes".to_string(),
        Type::Void => "void".to_string(),
        Type::List(inner) => format!("l_{}", mangle_type(inner)),
        Type::Map(k, v) => format!("m_{}_{}", mangle_type(k), mangle_type(v)),
        Type::Result(ok, err) => format!("r_{}_{}", mangle_type(ok), mangle_type(err)),
        Type::Option(inner) => format!("o_{}", mangle_type(inner)),
        Type::Async(inner) => format!("a_{}", mangle_type(inner)),
        Type::Channel(inner) => format!("ch_{}", mangle_type(inner)),
        Type::Array(inner, size) => format!("arr{}_{}", size, mangle_type(inner)),
        Type::Ref(inner) => format!("ref_{}", mangle_type(inner)),
        Type::Custom(name) => format!("c_{}", name),
        Type::Generic(name) => format!("g_{}", name),
    }
}

/// Stable C name of a generic function instantiated with the given substitution.
pub fn instance_c_name(fn_name: &str, type_params: &[String], subst: &HashMap<String, Type>) -> String {
    let m = type_params
        .iter()
        .map(|p| subst.get(p).map(mangle_type).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("_");
    format!("{}_{}", fn_name, m)
}