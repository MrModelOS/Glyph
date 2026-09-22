//! Shape & type checker — port of `ns/typechecker/shape_checker.cpp`.

use super::ast::{DimExpr, Dtype, Expr, ExprKind, Program, Stmt, StmtKind, TensorType, TypeNode};
use super::token::TokenType;
use std::collections::HashMap;

// Result of shape checking / inference
#[derive(Debug, Clone)]
pub struct ShapeError {
    pub message: String,
    pub line: u32,
    pub column: u32,
}

// Layer kinds with dimension inference rules
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Unknown,
    Dense,
    Linear,
    Dropout,
    LayerNorm,
    Attention,
    Embedding,
    MoE,
    Activation,
}

impl LayerKind {
    pub fn name(&self) -> &'static str {
        match self {
            LayerKind::Unknown => "Unknown",
            LayerKind::Dense => "Dense",
            LayerKind::Linear => "Linear",
            LayerKind::Dropout => "Dropout",
            LayerKind::LayerNorm => "LayerNorm",
            LayerKind::Attention => "Attention",
            LayerKind::Embedding => "Embedding",
            LayerKind::MoE => "MoE",
            LayerKind::Activation => "Activation",
        }
    }
}

// Inference rule: maps a layer's declared config params to input->output dims.
#[derive(Debug, Clone)]
pub struct LayerRule {
    pub kind: LayerKind,
    pub in_: DimExpr,        // Dense/Linear input features
    pub out: DimExpr,        // Dense/Linear output features
    pub emb_vocab: DimExpr,  // Embedding vocab
    pub emb_dim: DimExpr,    // Embedding dim
    pub num_heads: DimExpr,  // Attention
    pub num_experts: DimExpr, // MoE capacity (compile-time slot count)
    pub ffn_dim: DimExpr,    // MoE expert FFN hidden width (default 4*emb_dim)
    pub initial_experts: DimExpr, // MoE experts active at init
    pub activation: String,
    pub rate: f64, // Dropout rate
}

impl Default for LayerRule {
    fn default() -> Self {
        LayerRule {
            kind: LayerKind::Unknown,
            in_: DimExpr::dynamic(),
            out: DimExpr::dynamic(),
            emb_vocab: DimExpr::dynamic(),
            emb_dim: DimExpr::dynamic(),
            num_heads: DimExpr::dynamic(),
            num_experts: DimExpr::dynamic(),
            ffn_dim: DimExpr::dynamic(),
            initial_experts: DimExpr::dynamic(),
            activation: String::new(),
            rate: 0.0,
        }
    }
}

impl LayerRule {
    fn apply(&self, in_dims: &[DimExpr], ok: &mut bool) -> Vec<DimExpr> {
        *ok = false;
        let base: Vec<DimExpr> = Vec::new();
        match self.kind {
            LayerKind::Dense | LayerKind::Linear => {
                // [..., in_dim] -> [..., out_dim]
                *ok = true;
                if in_dims.is_empty() {
                    *ok = false;
                    return base;
                }
                let mut base = in_dims.to_vec();
                base.pop();
                base.push(self.out.clone());
                base
            }
            LayerKind::Dropout | LayerKind::LayerNorm | LayerKind::Activation => {
                // shape-preserving
                *ok = true;
                in_dims.to_vec()
            }
            LayerKind::Embedding => {
                // [B, seq] -> [B, seq, emb_dim]
                *ok = true;
                if in_dims.is_empty() {
                    *ok = false;
                    return base;
                }
                let mut base = in_dims.to_vec();
                base.push(self.emb_dim.clone());
                base
            }
            LayerKind::Attention => {
                // [B, seq, dim] keeps shape
                *ok = true;
                in_dims.to_vec()
            }
            LayerKind::MoE => {
                // output has the same shape as the input
                *ok = true;
                in_dims.to_vec()
            }
            LayerKind::Unknown => {
                *ok = false;
                base
            }
        }
    }
}

#[derive(Debug, Clone)]
struct FuncSig {
    param_types: Vec<TypeNode>,
    return_type: Option<TypeNode>,
}

#[derive(Debug, Clone, Default)]
struct SymbolBinding {
    bound_to_const: bool,
    const_value: i64,
    binds_to: String, // another symbolic name, if not const
}

pub struct ShapeChecker {
    errors_: Vec<ShapeError>,
    // Symbol table for type aliases: name -> DimExpr (alias constants)
    type_aliases_: HashMap<String, DimExpr>,
    // Symbol table for variables: name -> Type
    variables_: HashMap<String, TypeNode>,
    // Function signatures
    functions_: HashMap<String, FuncSig>,
    // Symbolic dimension constraint store
    symbols_: HashMap<String, SymbolBinding>,
    // Layer name -> inference rule
    layer_rules_: HashMap<String, LayerRule>,
}

impl ShapeChecker {
    pub fn new() -> Self {
        ShapeChecker {
            errors_: Vec::new(),
            type_aliases_: HashMap::new(),
            variables_: HashMap::new(),
            functions_: HashMap::new(),
            symbols_: HashMap::new(),
            layer_rules_: HashMap::new(),
        }
    }

    pub fn errors(&self) -> &[ShapeError] {
        &self.errors_
    }

    // Run full shape check on a program. Returns true if valid.
    pub fn check(&mut self, program: &mut Program) -> bool {
        self.errors_.clear();
        // First pass: collect type aliases
        for stmt in program.top_level.iter() {
            if stmt.kind == StmtKind::TypeDecl {
                if let Some(alias_expr) = &stmt.alias_expr {
                    if alias_expr.kind == ExprKind::LiteralInt {
                        let v: i64 = alias_expr.token.value.parse().unwrap_or(0);
                        self.type_aliases_.insert(stmt.alias_name.clone(), DimExpr::constant(v));
                    } else if alias_expr.kind == ExprKind::Identifier {
                        // Could be symbolic name (e.g., Dynamic) or ref to another alias
                        self.type_aliases_.insert(
                            stmt.alias_name.clone(),
                            DimExpr::symbolic(&alias_expr.token.value),
                        );
                    }
                }
            }
        }

        // Second pass: analyze all statements
        for stmt in program.top_level.iter_mut() {
            self.check_stmt(stmt);
        }
        self.errors_.is_empty()
    }

    pub fn get_expr_type(&self, expr: &Expr) -> TypeNode {
        if let Some(t) = &expr.inferred_type {
            (**t).clone()
        } else {
            TypeNode::scalar(Dtype::Float32)
        }
    }

    fn report(&mut self, line: u32, column: u32, msg: &str) {
        self.errors_.push(ShapeError {
            message: msg.to_string(),
            line,
            column,
        });
    }

    fn check_stmt(&mut self, stmt: &mut Stmt) {
        match stmt.kind {
            StmtKind::VarDecl => self.check_var_decl(stmt),
            StmtKind::FnDecl => self.check_fn_decl(stmt),
            StmtKind::NetworkDecl => self.check_network_decl(stmt),
            StmtKind::GradBlock => self.check_grad_block(stmt),
            StmtKind::Block => self.check_block(stmt),
            StmtKind::ReturnStmt => self.check_return(stmt),
            StmtKind::ExprStmt => self.check_expr_stmt(stmt),
            StmtKind::IfStmt => {
                if let Some(cond) = stmt.condition.as_mut() {
                    self.check_expr(cond);
                }
                if let Some(b) = stmt.body.as_mut() {
                    self.check_stmt(b);
                }
                if let Some(e) = stmt.else_branch.as_mut() {
                    self.check_stmt(e);
                }
            }
            StmtKind::WhileStmt => {
                if let Some(cond) = stmt.condition.as_mut() {
                    self.check_expr(cond);
                }
                if let Some(b) = stmt.body.as_mut() {
                    self.check_stmt(b);
                }
            }
            StmtKind::TypeDecl => {} // handled in first pass
            StmtKind::LayerDecl => {
                // Register layer rule so pipeline chains can resolve and apply it.
                let rule = self.parse_layer_rule(stmt);
                self.layer_rules_.insert(stmt.layer_name.clone(), rule);
            }
            StmtKind::ForwardDecl => {
                if let Some(b) = stmt.body.as_mut() {
                    self.check_stmt(b);
                }
            }
            StmtKind::TrainDecl => {
                if let Some(b) = stmt.body.as_mut() {
                    self.check_stmt(b);
                }
            }
            StmtKind::VarAssign => {
                if let Some(e) = stmt.init_expr.as_mut() {
                    self.check_expr(e);
                }
            }
            StmtKind::ImportStmt => {}
        }
    }

    fn check_var_decl(&mut self, stmt: &mut Stmt) {
        // Expand symbolic dims from type aliases
        if let Some(var_type) = stmt.var_type.as_mut() {
            if var_type.is_tensor() {
                for dim in var_type.tensor_type.dims.iter_mut() {
                    if dim.kind == super::ast::DimExprKind::Symbolic {
                        if let Some(alias) = self.type_aliases_.get(&dim.symbolic_name) {
                            if alias.is_const() {
                                *dim = alias.clone();
                            }
                        }
                        // Unknown symbolic stays symbolic (dependent type)
                    }
                }
            }
            self.variables_.insert(stmt.var_name.clone(), var_type.tensor_type.clone().into_type_node_scalar_check());
        }

        if let Some(init) = stmt.init_expr.as_mut() {
            self.check_expr(init);
            if init.inferred_type.is_some() && stmt.var_type.is_some() {
                // Type-check init expr against declared type
                let (var_tensor, var_line, var_col) = {
                    let vt = stmt.var_type.as_ref().unwrap();
                    (vt.tensor_type.clone(), stmt.token.line, stmt.token.column)
                };
                let init_tensor = init
                    .inferred_type
                    .as_ref()
                    .and_then(|t| if t.is_tensor() { Some(t.tensor_type.clone()) } else { None });
                if let Some(actual) = init_tensor {
                    if stmt.var_type.as_ref().unwrap().is_tensor() {
                        let mut declared = var_tensor;
                        if declared.dims.len() != actual.dims.len() {
                            let produced = actual.dims.len();
                            self.report(
                                var_line,
                                var_col,
                                &format!(
                                    "Dimension rank mismatch: declared {} but expression produces rank-{}",
                                    declared.to_string(),
                                    produced
                                ),
                            );
                        } else {
                            for i in 0..declared.dims.len() {
                                let decl_dim = declared.dims[i].clone();
                                let ok = self.unify_dim(&decl_dim, &actual.dims[i], var_line, var_col);
                                // Update variable's type with unified dim
                                if ok {
                                    declared.dims[i] = decl_dim;
                                }
                            }
                            if let Some(vt) = stmt.var_type.as_mut() {
                                vt.tensor_type.dims = declared.dims.clone();
                            }
                        }
                    }
                }
            }
            if stmt.var_type.is_some() {
                let vt = stmt.var_type.as_ref().unwrap();
                self.variables_.insert(stmt.var_name.clone(), vt.tensor_type.clone().into_type_node_scalar_check());
            } else if let Some(t) = &init.inferred_type {
                // Untyped declaration: bind to the inferred type of the init expr.
                self.variables_.insert(stmt.var_name.clone(), (**t).clone());
            }
        }
    }

    fn check_fn_decl(&mut self, stmt: &mut Stmt) {
        // Register function signature
        let mut sig = FuncSig {
            param_types: Vec::new(),
            return_type: None,
        };
        for p in &stmt.params {
            if let Some(t) = &p.type_ {
                sig.param_types.push((**t).clone());
            }
        }
        if let Some(t) = &stmt.return_type {
            sig.return_type = Some((**t).clone());
        }
        self.functions_.insert(stmt.fn_name.clone(), sig);

        // Push scope
        let saved = std::mem::take(&mut self.variables_);

        for p in &stmt.params {
            if let Some(t) = &p.type_ {
                self.variables_.insert(p.name.clone(), (**t).clone());
            }
        }

        if let Some(b) = stmt.body.as_mut() {
            self.check_block(b);
        }

        // Pop scope
        self.variables_ = saved;
    }

    fn check_network_decl(&mut self, stmt: &mut Stmt) {
        // Save global variable scope
        let saved = std::mem::take(&mut self.variables_);

        // inputs/outputs are vars
        for method in stmt.methods.iter_mut() {
            if method.kind == StmtKind::VarDecl {
                if let Some(var_type) = method.var_type.as_mut() {
                    let mut tt = var_type.tensor_type.clone();
                    for dim in tt.dims.iter_mut() {
                        self.bind_alias_dim(dim);
                        *dim = self.resolve_dim(dim);
                    }
                    self.variables_.insert(method.var_name.clone(), TypeNode::tensor(tt.clone()));
                    // refresh source var_type so declared input/output capture expanded dims
                    var_type.tensor_type = tt;
                }
            }
        }

        // Check layers (also register inference rules for pipeline propagation)
        let layer_names: Vec<String> = stmt.layers.iter().map(|l| l.layer_name.clone()).collect();
        for layer in stmt.layers.iter_mut() {
            let rule = self.parse_layer_rule(layer);
            self.layer_rules_.insert(layer.layer_name.clone(), rule.clone());
            self.check_stmt(layer);
            // Register the trainable weight matrix as a tensor variable.
            if (layer.layer_type == "Dense" || layer.layer_type == "Linear")
                && !self.variables_.contains_key(&layer.layer_name)
            {
                let uses_static = {
                    let r = self.layer_rules_.get(&layer.layer_name);
                    r.map(|r| r.in_.kind != super::ast::DimExprKind::Dynamic && r.out.kind != super::ast::DimExprKind::Dynamic)
                        .unwrap_or(false)
                };
                if uses_static {
                    let (din, dout) = {
                        let r = self.layer_rules_.get(&layer.layer_name).unwrap();
                        (r.in_.clone(), r.out.clone())
                    };
                    let mut din = din;
                    let mut dout = dout;
                    self.bind_alias_dim(&mut din);
                    self.bind_alias_dim(&mut dout);
                    if din.is_const() && dout.is_const() {
                        let mut wt = TensorType::new(Vec::new(), Dtype::Float32);
                        wt.dtype = Dtype::Float32;
                        wt.dims = vec![din, dout];
                        self.variables_.insert(layer.layer_name.clone(), TypeNode::tensor(wt));
                    }
                }
            }
        }
        let _ = layer_names;

        // Capture declared network input type (used to type forward params).
        let declared_input = stmt
            .methods
            .iter()
            .find(|m| m.kind == StmtKind::VarDecl && m.var_name == "input" && m.var_type.is_some())
            .map(|m| {
                let t = m.var_type.as_ref().unwrap();
                TypeNode::tensor(t.tensor_type.clone())
            });

        // Check forward
        for method in stmt.methods.iter_mut() {
            if method.kind == StmtKind::ForwardDecl {
                // Register forward params as tensor variables (typed by the input map)
                for p in method.params.iter_mut() {
                    if p.type_.is_none() {
                        // If no explicit type, default to this network's input type
                        if let Some(di) = &declared_input {
                            p.type_ = Some(Box::new(di.clone()));
                        } else if let Some(existing) = self.variables_.get(&p.name) {
                            p.type_ = Some(Box::new(existing.clone()));
                        } else {
                            p.type_ = Some(Box::new(TypeNode::scalar(Dtype::Float32)));
                        }
                    }
                    let t = p.type_.as_ref().unwrap();
                    self.variables_.insert(p.name.clone(), (**t).clone());
                }
            }
        }

        // Register train() params in scope
        for method in stmt.methods.iter_mut() {
            if method.kind != StmtKind::TrainDecl {
                continue;
            }
            for p in method.params.iter() {
                if let Some(t) = &p.type_ {
                    self.variables_.insert(p.name.clone(), (**t).clone());
                } else {
                    self.variables_.insert(p.name.clone(), TypeNode::scalar(Dtype::Float32));
                }
            }
        }

        // Capture declared network output type (for forward verification).
        let declared_output = stmt
            .methods
            .iter()
            .find(|m| m.kind == StmtKind::VarDecl && m.var_name == "output" && m.var_type.is_some())
            .map(|m| {
                let t = m.var_type.as_ref().unwrap();
                TypeNode::tensor(t.tensor_type.clone())
            });

        // Check the methods
        let mut methods: Vec<Box<Stmt>> = std::mem::take(&mut stmt.methods);
        for method in methods.iter_mut() {
            self.check_stmt(method);
        }

        // Verify the forward() return shape matches the declared output.
        if let Some(declared_output) = &declared_output {
            if declared_output.is_tensor() {
                for method in methods.iter_mut() {
                    if method.kind != StmtKind::ForwardDecl {
                        continue;
                    }
                    let rets: Vec<Box<Stmt>> = match method.body.as_mut() {
                        Some(b) => std::mem::take(&mut b.statements),
                        None => continue,
                    };
                    for s in rets.iter() {
                        if s.kind != StmtKind::ReturnStmt {
                            continue;
                        }
                        let init = match &s.init_expr {
                            Some(e) => e,
                            None => continue,
                        };
                        if let Some(t) = &init.inferred_type {
                            if t.is_tensor() {
                                let produced = t.tensor_type.clone();
                                let expected = declared_output.tensor_type.clone();
                                if produced.dims.len() != expected.dims.len() {
                                    self.report(
                                        stmt.token.line,
                                        stmt.token.column,
                                        &format!(
                                            "Network forward() produces rank-{} but declared output has rank-{}",
                                            produced.dims.len(),
                                            expected.dims.len()
                                        ),
                                    );
                                } else {
                                    for i in 0..expected.dims.len() {
                                        self.unify_dim(
                                            &expected.dims[i],
                                            &produced.dims[i],
                                            stmt.token.line,
                                            stmt.token.column,
                                        );
                                    }
                                }
                            }
                        }
                    }
                    // put statements back
                    if let Some(b) = method.body.as_mut() {
                        b.statements = rets;
                    }
                }
            }
        }
        stmt.methods = methods;

        self.variables_ = saved;
    }

    fn check_grad_block(&mut self, stmt: &mut Stmt) {
        if let Some(b) = stmt.grad_body.as_mut() {
            self.check_block(b);
        }
    }

    fn check_block(&mut self, stmt: &mut Stmt) {
        for s in stmt.statements.iter_mut() {
            self.check_stmt(s);
        }
    }

    fn check_return(&mut self, stmt: &mut Stmt) {
        if let Some(e) = stmt.init_expr.as_mut() {
            self.check_expr(e);
        }
    }

    fn check_expr_stmt(&mut self, stmt: &mut Stmt) {
        if let Some(e) = stmt.expr.as_mut() {
            self.check_expr(e);
        }
    }

    fn check_expr(&mut self, expr: &mut Expr) {
        match expr.kind {
            ExprKind::LiteralInt => {
                expr.inferred_type = Some(Box::new(TypeNode::scalar(Dtype::Int64)));
            }
            ExprKind::LiteralFloat => {
                expr.inferred_type = Some(Box::new(TypeNode::scalar(Dtype::Float64)));
            }
            ExprKind::LiteralBool => {
                expr.inferred_type = Some(Box::new(TypeNode::scalar(Dtype::Bool)));
            }
            ExprKind::LiteralString => {
                expr.inferred_type = Some(Box::new(TypeNode::scalar(Dtype::Int64))); // placeholder
            }
            ExprKind::Identifier => {
                let name = expr.token.value.clone();
                if let Some(vt) = self.lookup_var(&name) {
                    expr.inferred_type = Some(Box::new(vt));
                } else if self.layer_rules_.contains_key(&name) {
                    // Layer reference; treated as an opaque transform object.
                    expr.inferred_type = None;
                } else if name == "Dynamic" {
                    expr.inferred_type = None;
                } else {
                    self.report(expr.token.line, expr.token.column, &format!("Undefined variable '{}'", name));
                }
            }
            ExprKind::MatmulOp => {
                if let Some(l) = expr.left.as_mut() {
                    self.check_expr(l);
                }
                if let Some(r) = expr.right.as_mut() {
                    self.check_expr(r);
                }
                let (ll, rr) = {
                    let l = expr.left.as_ref().unwrap();
                    let r = expr.right.as_ref().unwrap();
                    (l.inferred_type.clone(), r.inferred_type.clone())
                };
                if let (Some(lt), Some(rt)) = (&ll, &rr) {
                    if lt.is_tensor() && rt.is_tensor() {
                        let (lhs, rhs) = (lt.tensor_type.clone(), rt.tensor_type.clone());
                        if self.matmul_compatible(&lhs, &rhs, expr.token.line, expr.token.column) {
                            let res = self.infer_matmul(&lhs, &rhs);
                            expr.inferred_type = Some(Box::new(TypeNode::tensor(res)));
                        }
                    }
                } else {
                    self.report(
                        expr.token.line,
                        expr.token.column,
                        "Matmul @ requires tensor operands",
                    );
                }
            }
            ExprKind::BinaryOp => {
                if let Some(l) = expr.left.as_mut() {
                    self.check_expr(l);
                }
                if let Some(r) = expr.right.as_mut() {
                    self.check_expr(r);
                }
                let t = expr.token.type_;
                let (ll, rr) = {
                    let l = expr.left.as_ref().unwrap();
                    let r = expr.right.as_ref().unwrap();
                    (l.inferred_type.clone(), r.inferred_type.clone())
                };
                if t == TokenType::OpAssign {
                    expr.inferred_type = ll;
                } else if ll.is_some() && rr.is_some() {
                    let lt = ll.as_ref().unwrap();
                    let rt = rr.as_ref().unwrap();
                    if lt.is_tensor() && rt.is_tensor() {
                        let (lhs, rhs) = (lt.tensor_type.clone(), rt.tensor_type.clone());
                        // Elementwise array op: same shapes
                        let mut ok = true;
                        if lhs.dims.len() == rhs.dims.len() {
                            for i in 0..lhs.dims.len() {
                                if !self.unify_dim(&lhs.dims[i], &rhs.dims[i], expr.token.line, expr.token.column) {
                                    ok = false;
                                }
                            }
                        } else {
                            ok = false;
                            self.report(
                                expr.token.line,
                                expr.token.column,
                                "Elementwise op shape rank mismatch",
                            );
                        }
                        if ok {
                            expr.inferred_type = Some(Box::new(TypeNode::tensor(lhs)));
                        }
                    } else if lt.is_scalar() {
                        expr.inferred_type = Some(Box::new((**lt).clone()));
                    } else {
                        expr.inferred_type = Some(Box::new(TypeNode::scalar(Dtype::Float32)));
                    }
                }
            }
            ExprKind::UnaryOp => {
                if let Some(o) = expr.operand.as_mut() {
                    self.check_expr(o);
                }
                if let Some(t) = &expr.operand.as_ref().and_then(|o| o.inferred_type.clone()) {
                    expr.inferred_type = Some(Box::new((**t).clone()));
                }
            }
            ExprKind::PipelineOp => {
                let upstream = {
                    // x -> fc1 -> drop -> fc2 : thread a tensor through layers.
                    // Walk the left-leaning tree to collect [...stages] names.
                    let mut stage_names: Vec<String> = Vec::new();
                    let mut cur: &mut Expr = expr;
                    loop {
                        if cur.kind == ExprKind::PipelineOp {
                            if let Some(r) = cur.right.as_ref() {
                                stage_names.push(r.token.value.clone());
                            }
                            match cur.left.as_mut() {
                                Some(l) => cur = l.as_mut(),
                                None => {
                                    // no input tensor
                                    self.report(
                                        expr.token.line,
                                        expr.token.column,
                                        "Invalid pipeline with no input tensor",
                                    );
                                    return;
                                }
                            }
                        } else {
                            break;
                        }
                    }
                    stage_names.reverse();

                    // Resolve the input tensor type.
                    self.check_expr(cur);
                    let input_type = cur.inferred_type.clone();
                    (stage_names, input_type)
                };

                let (stage_names, input_type) = upstream;
                let mut current_type: Option<TensorType> = None;
                if let Some(t) = &input_type {
                    if t.is_tensor() {
                        current_type = Some(t.tensor_type.clone());
                    }
                }

                // Apply each stage in order.
                if let Some(mut ct) = current_type {
                    for name in &stage_names {
                        if let Some(rule) = self.layer_rules_.get(name).cloned() {
                            let mut next = TensorType::new(Vec::new(), Dtype::Float32);
                            self.apply_layer(&rule, &ct, &mut next, expr.token.line, expr.token.column);
                            ct = next;
                        }
                        // Unknown stage: treat as shape-preserving passthrough.
                    }
                    current_type = Some(self.resolve_tensor(&ct));
                }

                if current_type.is_some() {
                    expr.inferred_type = Some(Box::new(TypeNode::tensor(current_type.unwrap())));
                }
            }
            ExprKind::FunctionCall => {
                // Self-contained argument check
                let args: Vec<Box<Expr>> = std::mem::take(&mut expr.args);
                for mut a in args {
                    self.check_expr(&mut a);
                    expr.args.push(a);
                }
                let func_name = expr
                    .operand
                    .as_ref()
                    .map(|o| if o.kind == ExprKind::Identifier { o.token.value.clone() } else { String::new() })
                    .unwrap_or_default();
                match func_name.as_str() {
                    "cross_entropy" => {
                        // loss is scalar
                        expr.inferred_type = Some(Box::new(TypeNode::scalar(Dtype::Float32)));
                        if expr.args.len() >= 2 {
                            let (a0, a1) = {
                                let a0 = expr.args[0].as_ref();
                                let a1 = expr.args[1].as_ref();
                                (
                                    a0.inferred_type.as_ref().map(|t| t.tensor_type.clone()),
                                    a1.inferred_type.as_ref().map(|t| t.tensor_type.clone()),
                                )
                            };
                            if let (Some(preds), Some(labels)) = (a0, a1) {
                                if preds.dims.len() != 2 || labels.dims.len() != 2 {
                                    self.report(
                                        expr.token.line,
                                        expr.token.column,
                                        "cross_entropy expects 2D tensors [B, C]",
                                    );
                                } else {
                                    self.unify_dim(
                                        &preds.dims[0],
                                        &labels.dims[0],
                                        expr.token.line,
                                        expr.token.column,
                                    );
                                    self.unify_dim(
                                        &preds.dims[1],
                                        &labels.dims[1],
                                        expr.token.line,
                                        expr.token.column,
                                    );
                                }
                            }
                        }
                    }
                    "Dense" | "Linear" | "Dropout" | "LayerNorm" | "Attention" | "Embedding" => {
                        // Layer constructors produce a layer object; infer minimal
                        expr.inferred_type = Some(Box::new(TypeNode::scalar(Dtype::Float32)));
                    }
                    "relu" | "leaky_relu" | "sigmoid" | "tanh" | "silu" | "swish" | "gelu"
                    | "softmax" | "identity" | "dropout" => {
                        // Elementwise activation: shape-preserving passthrough.
                        if let Some(a) = expr.args.first() {
                            if let Some(t) = &a.inferred_type {
                                expr.inferred_type = Some(Box::new((**t).clone()));
                            } else {
                                expr.inferred_type = Some(Box::new(TypeNode::scalar(Dtype::Float32)));
                            }
                        } else {
                            expr.inferred_type = Some(Box::new(TypeNode::scalar(Dtype::Float32)));
                        }
                    }
                    "slice" | "index" | "scatter" | "concat" | "transpose" | "reshape" => {
                        // v1.2 data-movement builtins.
                        if let Some(a) = expr.args.first() {
                            if let Some(t) = &a.inferred_type {
                                if t.is_tensor() {
                                    let mut tt = t.tensor_type.clone();
                                    let fname = func_name.as_str();
                                    if fname == "transpose" && tt.dims.len() == 2 {
                                        tt.dims.swap(0, 1);
                                    } else if fname == "concat" && expr.args.len() >= 2 {
                                        let tb = expr.args[1]
                                            .as_ref()
                                            .inferred_type
                                            .as_ref()
                                            .map(|t| t.tensor_type.clone());
                                        if let Some(tb) = tb {
                                            if tt.dims.len() == 2 && tb.dims.len() == 2 {
                                                let mut axis: i64 = 1;
                                                if expr.args.len() >= 3
                                                    && expr.args[2].kind == ExprKind::LiteralInt
                                                {
                                                    axis = expr.args[2].token.value.parse().unwrap_or(1);
                                                }
                                                if axis == 0 {
                                                    let a = tt.dims[0].const_value;
                                                    let b = tb.dims[0].const_value;
                                                    tt.dims[0] = DimExpr::constant(a + b);
                                                } else {
                                                    let ca = tt.dims[1].const_value;
                                                    let cb = tb.dims[1].const_value;
                                                    tt.dims[1] = DimExpr::constant(ca + cb);
                                                }
                                            }
                                        }
                                    } else if fname == "slice"
                                        && tt.dims.len() == 2
                                        && expr.args.len() >= 4
                                        && expr.args[2].kind == ExprKind::LiteralInt
                                        && expr.args[3].kind == ExprKind::LiteralInt
                                    {
                                        let s: i64 = expr.args[2].token.value.parse().unwrap_or(0);
                                        let e: i64 = expr.args[3].token.value.parse().unwrap_or(0);
                                        let mut axis: i64 = 1;
                                        if expr.args[1].kind == ExprKind::LiteralInt {
                                            axis = expr.args[1].token.value.parse().unwrap_or(1);
                                        }
                                        tt.dims[if axis == 0 { 0 } else { 1 }] = DimExpr::constant(e - s);
                                    } else if fname == "index" && tt.dims.len() == 2 {
                                        let mut axis: i64 = 1;
                                        if expr.args.len() >= 2 && expr.args[1].kind == ExprKind::LiteralInt {
                                            axis = expr.args[1].token.value.parse().unwrap_or(1);
                                        }
                                        let nidx = expr.args.len() as i64 - 2;
                                        if axis == 0 {
                                            tt.dims[0] = DimExpr::constant(nidx);
                                        } else {
                                            tt.dims[1] = DimExpr::constant(nidx);
                                        }
                                    }
                                    expr.inferred_type = Some(Box::new(TypeNode::tensor(tt)));
                                }
                            } else {
                                expr.inferred_type = Some(Box::new(TypeNode::scalar(Dtype::Float32)));
                            }
                        } else {
                            expr.inferred_type = Some(Box::new(TypeNode::scalar(Dtype::Float32)));
                        }
                    }
                    "forward" | "step" => {
                        // Network method call; result is a tensor (passthrough).
                        if let Some(a) = expr.args.first() {
                            if let Some(t) = &a.inferred_type {
                                expr.inferred_type = Some(Box::new((**t).clone()));
                            } else {
                                expr.inferred_type = Some(Box::new(TypeNode::scalar(Dtype::Float32)));
                            }
                        } else {
                            expr.inferred_type = Some(Box::new(TypeNode::scalar(Dtype::Float32)));
                        }
                    }
                    _ => {
                        if let Some(sig) = self.functions_.get(&func_name).cloned() {
                            if let Some(rt) = sig.return_type {
                                expr.inferred_type = Some(Box::new(rt));
                            } else {
                                expr.inferred_type = Some(Box::new(TypeNode::scalar(Dtype::Float32)));
                            }
                        } else {
                            self.report(
                                expr.token.line,
                                expr.token.column,
                                &format!("Unknown function '{}'", func_name),
                            );
                            expr.inferred_type = Some(Box::new(TypeNode::scalar(Dtype::Float32)));
                        }
                    }
                }
            }
            ExprKind::IndexOp => {
                if let Some(o) = expr.operand.as_mut() {
                    self.check_expr(o);
                }
                let indices: Vec<Box<Expr>> = std::mem::take(&mut expr.indices);
                for mut idx in indices {
                    self.check_expr(&mut idx);
                    expr.indices.push(idx);
                }
                if let Some(t) = &expr.operand.as_ref().and_then(|o| o.inferred_type.clone()) {
                    if t.is_tensor() && !expr.indices.is_empty() {
                        // Reduce dim by index count -> scalar or smaller tensor
                        expr.inferred_type = Some(Box::new(TypeNode::scalar(Dtype::Float32)));
                    }
                }
            }
            ExprKind::TypeCast => {
                if let Some(o) = expr.operand.as_mut() {
                    self.check_expr(o);
                }
                if let Some(t) = &expr.operand.as_ref().and_then(|o| o.inferred_type.clone()) {
                    expr.inferred_type = Some(Box::new((**t).clone()));
                }
            }
            ExprKind::TensorLiteral => {
                expr.inferred_type = Some(Box::new(TypeNode::scalar(Dtype::Float32)));
            }
        }
    }

    fn matmul_compatible(
        &mut self,
        lhs: &TensorType,
        rhs: &TensorType,
        line: u32,
        col: u32,
    ) -> bool {
        if lhs.dims.len() != 2 || rhs.dims.len() != 2 {
            self.report(
                line,
                col,
                &format!(
                    "Matmul @ requires 2D tensors, got {}D and {}D",
                    lhs.dims.len(),
                    rhs.dims.len()
                ),
            );
            return false;
        }
        // lhs: [A, B], rhs: [C, D] -> requires B == C
        let inner_lhs = &lhs.dims[1];
        let inner_rhs = &rhs.dims[0];

        if !self.unify_dim(inner_lhs, inner_rhs, line, col) {
            return false;
        }
        // Report informative message if mismatch
        if inner_lhs.is_const() && inner_rhs.is_const() && inner_lhs.const_value != inner_rhs.const_value {
            self.report(
                line,
                col,
                &format!(
                    "Shape mismatch in @: inner dims {} vs {}",
                    inner_lhs.const_value, inner_rhs.const_value
                ),
            );
            return false;
        }
        true
    }

    fn infer_matmul(&mut self, lhs: &TensorType, rhs: &TensorType) -> TensorType {
        let mut result = TensorType::new(Vec::new(), Dtype::Float32);
        result.dims.push(lhs.dims[0].clone()); // row dim from lhs
        result.dims.push(rhs.dims[1].clone()); // col dim from rhs
        result.dtype = lhs.dtype;
        result
    }

    fn unify_dim(&mut self, a: &DimExpr, b: &DimExpr, line: u32, col: u32) -> bool {
        // Dynamic unifies with anything.
        if a.is_dynamic() || b.is_dynamic() {
            return true;
        }

        if a.is_const() && b.is_const() {
            if a.const_value == b.const_value {
                return true;
            }
            self.report(
                line,
                col,
                &format!(
                    "Dimension mismatch: constant {} vs {}",
                    a.const_value, b.const_value
                ),
            );
            return false;
        }

        // At least one side is symbolic.
        if a.is_symbolic() && b.is_symbolic() {
            if a.symbolic_name == b.symbolic_name {
                return true;
            }
            self.merge_symbols(&a.symbolic_name, &b.symbolic_name);
            return true;
        }
        if a.is_symbolic() && b.is_const() {
            self.record_symbol_const(&a.symbolic_name, b.const_value);
            return true;
        }
        if a.is_const() && b.is_symbolic() {
            self.record_symbol_const(&b.symbolic_name, a.const_value);
            return true;
        }
        true
    }

    fn record_symbol_const(&mut self, name: &str, value: i64) {
        let sb = self.symbols_.entry(name.to_string()).or_default();
        if sb.bound_to_const {
            return; // keep the first binding
        }
        sb.bound_to_const = true;
        sb.const_value = value;
        sb.binds_to.clear();
    }

    fn merge_symbols(&mut self, a: &str, b: &str) {
        if a == b {
            return;
        }
        let a_binding = self.symbols_.get(a).map(|s| (s.bound_to_const, s.const_value)).clone();
        let b_binding = self.symbols_.get(b).map(|s| (s.bound_to_const, s.const_value)).clone();
        let a_binds: Option<String> = self.symbols_.get(a).map(|s| s.binds_to.clone()).clone();
        // If one side is already bound to a constant, propagate it.
        if let Some((true, v)) = a_binding {
            self.record_symbol_const(b, v);
            return;
        }
        if let Some((true, v)) = b_binding {
            self.record_symbol_const(a, v);
            return;
        }
        // Otherwise chain them.
        match a_binds {
            Some(binds) if !binds.is_empty() => {
                let b = b.to_string();
                self.merge_symbols(&binds, &b);
            }
            _ => {
                let sa = self.symbols_.entry(a.to_string()).or_default();
                sa.binds_to = b.to_string();
            }
        }
    }

    fn resolve_dim(&self, d: &DimExpr) -> DimExpr {
        if !d.is_symbolic() {
            return d.clone();
        }
        // Follow bindings
        let mut cur = d.symbolic_name.clone();
        let mut seen = std::collections::HashSet::new();
        loop {
            if seen.contains(&cur) {
                break;
            }
            seen.insert(cur.clone());
            match self.symbols_.get(&cur) {
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

    fn resolve_tensor(&self, t: &TensorType) -> TensorType {
        let mut r = t.clone();
        for dim in r.dims.iter_mut() {
            *dim = self.resolve_dim(dim);
        }
        r
    }

    fn bind_alias_dim(&mut self, dim: &mut DimExpr) {
        if dim.kind == super::ast::DimExprKind::Symbolic {
            if let Some(alias) = self.type_aliases_.get(&dim.symbolic_name).cloned() {
                if alias.is_const() {
                    *dim = alias;
                } // symbolic alias keeps symbolic
            }
        }
    }

    fn parse_layer_rule(&self, layer: &Stmt) -> LayerRule {
        let mut rule = LayerRule::default();
        let lt = layer.layer_type.as_str();
        rule.kind = match lt {
            "Dense" | "Linear" => LayerKind::Dense,
            "Dropout" => LayerKind::Dropout,
            "LayerNorm" => LayerKind::LayerNorm,
            "Attention" | "MultiHeadAttention" => LayerKind::Attention,
            "Embedding" => LayerKind::Embedding,
            "MoE" | "MixtureOfExperts" => LayerKind::MoE,
            _ => LayerKind::Unknown,
        };

        for p in &layer.layer_params {
            let v: String = p
                .value
                .as_ref()
                .map(|e| e.token.value.clone())
                .unwrap_or_default();
            let is_num = v.as_bytes().first().map(|b| b.is_ascii_digit()).unwrap_or(false);
            match p.name.as_str() {
                "in" | "in_features" => {
                    if is_num {
                        rule.in_ = DimExpr::constant(v.parse().unwrap_or(0));
                    } else {
                        rule.in_ = DimExpr::symbolic(&v);
                    }
                }
                "out" | "out_features" => {
                    if is_num {
                        rule.out = DimExpr::constant(v.parse().unwrap_or(0));
                    } else {
                        rule.out = DimExpr::symbolic(&v);
                    }
                }
                "activation" | "act" => rule.activation = v,
                "rate" | "p" => {
                    if is_num {
                        rule.rate = v.parse().unwrap_or(0.0);
                    }
                }
                "vocab_size" => {
                    if is_num {
                        rule.emb_vocab = DimExpr::constant(v.parse().unwrap_or(0));
                    } else {
                        rule.emb_vocab = DimExpr::symbolic(&v);
                    }
                }
                "d_model" | "n_embd" | "dim" => {
                    if is_num {
                        rule.emb_dim = DimExpr::constant(v.parse().unwrap_or(0));
                    } else {
                        rule.emb_dim = DimExpr::symbolic(&v);
                    }
                }
                "heads" | "num_heads" => {
                    if is_num {
                        rule.num_heads = DimExpr::constant(v.parse().unwrap_or(0));
                    } else {
                        rule.num_heads = DimExpr::symbolic(&v);
                    }
                }
                "experts" | "num_experts" => {
                    if is_num {
                        rule.num_experts = DimExpr::constant(v.parse().unwrap_or(0));
                    } else {
                        rule.num_experts = DimExpr::symbolic(&v);
                    }
                }
                "ffn_dim" | "ffn" | "hidden" | "d_ff" | "intermediate" => {
                    if is_num {
                        rule.ffn_dim = DimExpr::constant(v.parse().unwrap_or(0));
                    } else {
                        rule.ffn_dim = DimExpr::symbolic(&v);
                    }
                }
                "initial_experts" | "active" | "k" | "initial_active" => {
                    if is_num {
                        rule.initial_experts = DimExpr::constant(v.parse().unwrap_or(0));
                    } else {
                        rule.initial_experts = DimExpr::symbolic(&v);
                    }
                }
                _ => {}
            }
        }
        rule
    }

    fn apply_layer(
        &mut self,
        rule: &LayerRule,
        input: &TensorType,
        out: &mut TensorType,
        line: u32,
        col: u32,
    ) {
        let mut ok = false;
        let out_dims = rule.apply(&input.dims, &mut ok);
        if !ok {
            self.report(
                line,
                col,
                &format!("Layer cannot be applied to input of rank {}", input.dims.len()),
            );
            *out = input.clone();
            return;
        }
        out.dims = out_dims;
        out.dtype = input.dtype;
        // Resolve any symbolic choices introduced by the rule
        let mut resolved: Vec<DimExpr> = Vec::new();
        for d in &out.dims {
            let r1 = self.resolve_dim(d);
            let mut r2 = r1;
            self.bind_alias_dim(&mut r2);
            resolved.push(self.resolve_dim(&r2));
        }
        out.dims = resolved;
    }

    fn lookup_var(&self, name: &str) -> Option<TypeNode> {
        self.variables_.get(name).cloned()
    }
}

// Small adapter so `TensorType` can be turned into a `TypeNode` in two spots
// (keeps the main body mirroring the C++).
trait TypeNodeFromTensor {
    fn into_type_node_scalar_check(self) -> TypeNode;
}

impl TypeNodeFromTensor for TensorType {
    fn into_type_node_scalar_check(self) -> TypeNode {
        TypeNode::tensor(self)
    }
}