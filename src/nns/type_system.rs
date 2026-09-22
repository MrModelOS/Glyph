#![allow(dead_code)]
//! Type-system helpers — port of `ns/typechecker/type_system.cpp`.

use std::collections::{HashMap, HashSet};

use super::ast::{DimExpr, Dtype};

/// Highest-precedence dtype promotion for elementwise binary ops.
/// e.g. float16 + float32 -> float32 (and so on).
pub fn promote_dtype(a: Dtype, b: Dtype) -> Dtype {
    // Simple ranking: Bool < Int-ish < FP8 < FP16 < FP32 < FP64
    fn rank(d: Dtype) -> i32 {
        match d {
            Dtype::Bool => 0,
            Dtype::Int8 => 1,
            Dtype::Int16 => 2,
            Dtype::Int32 => 3,
            Dtype::Int64 => 4,
            Dtype::Fp4 => 5,
            Dtype::Fp8 => 6,
            Dtype::Float16 => 7,
            Dtype::Float32 => 8,
            Dtype::Float64 => 9,
        }
    }
    if rank(a) >= rank(b) {
        a
    } else {
        b
    }
}

/// Symbolic dimension binding — mirrors `ShapeChecker::SymbolBinding`.
#[derive(Debug, Clone, Default)]
pub struct SymbolBinding {
    pub bound_to_const: bool,
    pub const_value: i64,
    pub binds_to: String,
}

/// Resolve a symbolic dimension name via the type-alias table.
///
/// If `name` is an alias for a constant dimension (e.g. `type Hidden = 16`),
/// returns `Some(const)`. If the alias itself is symbolic (e.g. `type A = B`
/// where `B` is still symbolic), the alias chain is followed until a concrete
/// constant or an unknown symbolic is reached. Returns `None` if `name` has no
/// alias entry at all.
pub fn resolve_dim_symbolic(name: &str, aliases: &HashMap<String, DimExpr>) -> Option<DimExpr> {
    let first = aliases.get(name)?.clone();
    if first.is_const() {
        return Some(first);
    }
    if first.is_dynamic() {
        return Some(first);
    }
    // Symbolic alias — follow chain with cycle detection.
    let mut cur = first.symbolic_name.clone();
    let mut seen: HashSet<String> = HashSet::new();
    seen.insert(name.to_string());
    loop {
        if seen.contains(&cur) {
            // cycle: return the symbolic dim that caused it
            return Some(DimExpr::symbolic(&cur));
        }
        seen.insert(cur.clone());
        match aliases.get(&cur) {
            Some(next) if next.is_const() => return Some(next.clone()),
            Some(next) if next.is_symbolic() => {
                cur = next.symbolic_name.clone();
                continue;
            }
            Some(next) if next.is_dynamic() => return Some(next.clone()),
            Some(_) | None => break,
        }
    }
    Some(DimExpr::symbolic(&cur))
}

/// Resolve a symbolic dimension name via the symbolic-constraint table
/// (`symbols` built by `ShapeChecker::unify_dim`).
///
/// Follows `binds_to` chains and returns a constant if the name was bound to
/// one, otherwise returns the terminal symbolic name.
pub fn resolve_symbolic_via_bindings(name: &str, symbols: &HashMap<String, SymbolBinding>) -> DimExpr {
    let mut cur = name.to_string();
    let mut seen: HashSet<String> = HashSet::new();
    loop {
        if seen.contains(&cur) {
            break;
        }
        seen.insert(cur.clone());
        match symbols.get(&cur) {
            None => break,
            Some(sb) => {
                if sb.bound_to_const {
                    return DimExpr::constant(sb.const_value);
                }
                if sb.binds_to.is_empty() {
                    break;
                }
                cur = sb.binds_to.clone();
            }
        }
    }
    DimExpr::symbolic(&cur)
}

/// Convenience: resolve a `DimExpr` (if symbolic) via aliases, returning the
/// resolved `DimExpr` or the original if no alias applies.
pub fn resolve_dim_with_aliases(dim: &DimExpr, aliases: &HashMap<String, DimExpr>) -> DimExpr {
    if !dim.is_symbolic() {
        return dim.clone();
    }
    if let Some(resolved) = resolve_dim_symbolic(&dim.symbolic_name, aliases) {
        // Only replace if the alias resolved to a constant — a symbolic alias
        // that stays symbolic is kept as-is (dependent type).
        if resolved.is_const() || resolved.is_dynamic() {
            return resolved;
        }
        // If alias was symbolic and chain ended on another symbolic, keep that.
        // But if the alias was a pure symbolic forwarding (e.g. A -> B) we
        // return the terminal symbolic so callers see the canonical name.
        return resolved;
    }
    dim.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_dim_symbolic_const_alias() {
        let mut aliases = HashMap::new();
        aliases.insert("Hidden".to_string(), DimExpr::constant(16));
        let r = resolve_dim_symbolic("Hidden", &aliases).unwrap();
        assert!(r.is_const());
        assert_eq!(r.const_value, 16);
    }

    #[test]
    fn resolve_dim_symbolic_missing_returns_none() {
        let aliases = HashMap::new();
        assert!(resolve_dim_symbolic("Missing", &aliases).is_none());
    }

    #[test]
    fn resolve_dim_symbolic_chain() {
        let mut aliases = HashMap::new();
        aliases.insert("A".to_string(), DimExpr::symbolic("B"));
        aliases.insert("B".to_string(), DimExpr::constant(32));
        let r = resolve_dim_symbolic("A", &aliases).unwrap();
        assert!(r.is_const());
        assert_eq!(r.const_value, 32);
    }

    #[test]
    fn resolve_dim_with_aliases_passthrough() {
        let aliases = HashMap::new();
        let d = DimExpr::symbolic("Unknown");
        let r = resolve_dim_with_aliases(&d, &aliases);
        assert!(r.is_symbolic());
        assert_eq!(r.symbolic_name, "Unknown");
    }

    #[test]
    fn resolve_symbolic_via_bindings_const() {
        let mut symbols = HashMap::new();
        symbols.insert(
            "N".to_string(),
            SymbolBinding {
                bound_to_const: true,
                const_value: 64,
                binds_to: String::new(),
            },
        );
        let r = resolve_symbolic_via_bindings("N", &symbols);
        assert!(r.is_const());
        assert_eq!(r.const_value, 64);
    }

    #[test]
    fn resolve_symbolic_via_bindings_chain() {
        let mut symbols = HashMap::new();
        symbols.insert(
            "A".to_string(),
            SymbolBinding {
                bound_to_const: false,
                const_value: 0,
                binds_to: "B".to_string(),
            },
        );
        symbols.insert(
            "B".to_string(),
            SymbolBinding {
                bound_to_const: true,
                const_value: 128,
                binds_to: String::new(),
            },
        );
        let r = resolve_symbolic_via_bindings("A", &symbols);
        assert!(r.is_const());
        assert_eq!(r.const_value, 128);
    }
}