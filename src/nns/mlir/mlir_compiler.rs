//! MLIR compiler — port of `ns/mlir/mlir_compiler.cpp`.
//!
//! Converts the NS AST to an MLIR module (high-level tensor IR).

use std::collections::HashMap;

use super::super::ast::{DimExpr, Dtype, Expr, ExprKind, Program, Stmt, StmtKind, TensorType};
use super::dialect::*;

/// MLIR compiler — converts NS AST into the ns.tensor dialect IR.
pub struct MLIRCompiler {
    temp_counter: usize,
    /// Variable name -> produced value id, scoped per compiled function body.
    local_var_ids: HashMap<String, String>,
    /// Top-level `type X = <const|alias>` aliases, for resolving layer dims.
    aliases: HashMap<String, i64>,
    /// Layer name -> weight buffer id (network layer params map onto *_w ids).
    layer_weight_id: HashMap<String, String>,
    /// Layer metadata for forward lowering (activation, dropout rate).
    layer_meta: HashMap<String, LayerMeta>,
}

#[derive(Debug, Clone)]
struct LayerMeta {
    type_name: String,       // "Dense", "Dropout", "Embedding", ...
    activation: String,      // "ReLU"/"GELU"/... (empty if none)
    dropout_rate: f64,
    num_heads: i64,
    causal: bool,
    num_experts: i64,
    ffn_dim: i64,
    initial_experts: i64,
    emb_dim: i64,
    vocab_size: i64,
    // Weight buffer ids for layers with >1 trainable matrices.
    weight_ids: Vec<String>,
}

impl Default for LayerMeta {
    fn default() -> Self {
        LayerMeta {
            type_name: String::new(),
            activation: String::new(),
            dropout_rate: 0.0,
            num_heads: 1,
            causal: false,
            num_experts: 4,
            ffn_dim: 0,
            initial_experts: 0,
            emb_dim: 0,
            vocab_size: 0,
            weight_ids: Vec::new(),
        }
    }
}

impl MLIRCompiler {
    pub fn new() -> Self {
        MLIRCompiler {
            temp_counter: 0,
            local_var_ids: HashMap::new(),
            aliases: HashMap::new(),
            layer_weight_id: HashMap::new(),
            layer_meta: HashMap::new(),
        }
    }

    fn new_temp(&mut self, prefix: &str) -> String {
        let r = format!("{}{}", prefix, self.temp_counter);
        self.temp_counter += 1;
        r
    }

    fn reset_locals(&mut self) {
        self.local_var_ids.clear();
    }

    fn resolve_dim_int(&self, expr: &Expr, out: &mut i64) -> bool {
        match expr.kind {
            ExprKind::LiteralInt => {
                if let Ok(v) = expr.token.value.parse::<i64>() {
                    *out = v;
                    true
                } else {
                    false
                }
            }
            ExprKind::LiteralFloat => {
                if let Ok(v) = expr.token.value.parse::<i64>() {
                    *out = v;
                    true
                } else {
                    false
                }
            }
            ExprKind::Identifier => {
                if let Some(&v) = self.aliases.get(&expr.token.value) {
                    *out = v;
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn activation_name_from_op(op: MLIROp) -> &'static str {
        match op {
            MLIROp::Relu => "relu",
            MLIROp::LeakyRelu => "leaky_relu",
            MLIROp::Sigmoid => "sigmoid",
            MLIROp::Tanh => "tanh",
            MLIROp::Swish => "swish",
            MLIROp::Gelu => "gelu",
            MLIROp::Silu => "silu",
            MLIROp::Identity => "identity",
            MLIROp::Softmax => "softmax",
            MLIROp::Layernorm => "layernorm",
            _ => "identity",
        }
    }

    fn activation_op_from_name(act: &str) -> MLIROp {
        match act {
            "ReLU" => MLIROp::Relu,
            "Sigmoid" => MLIROp::Sigmoid,
            "Tanh" => MLIROp::Tanh,
            "GELU" => MLIROp::Gelu,
            "Swish" | "SiLU" => MLIROp::Swish,
            "LeakyReLU" => MLIROp::LeakyRelu,
            "Softmax" => MLIROp::Softmax,
            _ => MLIROp::Identity,
        }
    }

    fn set_grad(
        &mut self,
        g: &mut HashMap<String, String>,
        var: &str,
        grad_id: &str,
        fn_: &mut MLIRFunction,
    ) {
        if let Some(existing) = g.get(var) {
            if !existing.is_empty() && existing != grad_id {
                // Accumulate: emit elementwise add of existing + new grad.
                let sum_id = self.new_temp("g");
                let mut sum = MLIRInstr::new(MLIROp::ElementwiseBinop, sum_id);
                sum.operands = vec![existing.clone(), grad_id.to_string()];
                sum.attribute = "+".to_string();
                sum.comment = format!("accumulate d{}", var);
                let sum_id_out = sum.result_id.clone();
                fn_.instructions.push(sum);
                g.insert(var.to_string(), sum_id_out);
                return;
            }
        }
        g.insert(var.to_string(), grad_id.to_string());
    }

    // -------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------

    /// Convert NS AST to MLIR module.
    pub fn compile(&mut self, program: &mut Program) -> MLIRModule {
        self.collect_aliases(program);
        let mut module = MLIRModule::default();
        let mut main_fn = MLIRFunction::default();
        main_fn.name = "main".to_string();
        let mut main_has_content = false;

        for stmt in program.top_level.iter() {
            match stmt.kind {
                StmtKind::FnDecl => {
                    self.compile_fn(stmt, &mut module);
                }
                StmtKind::NetworkDecl => {
                    self.compile_network(stmt, &mut module);
                }
                StmtKind::VarDecl | StmtKind::VarAssign | StmtKind::ExprStmt | StmtKind::ReturnStmt => {
                    self.compile_stmt(stmt, &mut main_fn);
                    main_has_content = true;
                }
                _ => {}
            }
        }
        if main_has_content {
            module.functions.push(main_fn);
        }
        module
    }

    /// Pretty-print MLIR text representation.
    pub fn dump(module: &MLIRModule) -> String {
        let mut out = String::from("// NeuralScript MLIR\n");
        for fn_ in &module.functions {
            out.push_str(&format!("ns.func @{} {{\n", fn_.name));
            for instr in &fn_.instructions {
                out.push_str("  ");
                let op_name = match instr.op {
                    MLIROp::TensorAlloc => "ns.tensor.alloc",
                    MLIROp::TensorFree => "ns.tensor.free",
                    MLIROp::Matmul => "ns.matmul",
                    MLIROp::ElementwiseBinop => "ns.subview_binop",
                    MLIROp::Relu => "ns.activation.relu",
                    MLIROp::LeakyRelu => "ns.activation.leaky_relu",
                    MLIROp::Sigmoid => "ns.activation.sigmoid",
                    MLIROp::Tanh => "ns.activation.tanh",
                    MLIROp::Swish => "ns.activation.swish",
                    MLIROp::Gelu => "ns.activation.gelu",
                    MLIROp::Silu => "ns.activation.silu",
                    MLIROp::Identity => "ns.activation.identity",
                    MLIROp::Softmax => "ns.activation.softmax",
                    MLIROp::Dropout => "ns.dropout",
                    MLIROp::Layernorm => "ns.layernorm",
                    MLIROp::CrossEntropy => "ns.cross_entropy",
                    MLIROp::LayernormGrad => "ns.grad.layernorm",
                    MLIROp::EmbeddingGradW => "ns.grad.embedding.w",
                    MLIROp::MoeGradX => "ns.grad.moe.x",
                    MLIROp::MoeGradWg => "ns.grad.moe.wg",
                    MLIROp::MoeGradWe1 => "ns.grad.moe.we1",
                    MLIROp::MoeGradWe2 => "ns.grad.moe.we2",
                    MLIROp::AttentionGradX => "ns.grad.attention.x",
                    MLIROp::AttentionGradWq => "ns.grad.attention.wq",
                    MLIROp::AttentionGradWk => "ns.grad.attention.wk",
                    MLIROp::AttentionGradWv => "ns.grad.attention.wv",
                    MLIROp::AttentionGradWo => "ns.grad.attention.wo",
                    MLIROp::MatmulGradA => "ns.grad.matmul.a",
                    MLIROp::MatmulGradW => "ns.grad.matmul.w",
                    MLIROp::ActivationGrad => "ns.grad.activation",
                    MLIROp::LossGrad => "ns.grad.loss",
                    MLIROp::BinopGrad => "ns.grad.binop",
                    MLIROp::Grad => "ns.grad",
                    MLIROp::Forward => "ns.forward",
                    MLIROp::LayerDense => "ns.layer.dense",
                    MLIROp::LayerDropout => "ns.layer.dropout",
                    MLIROp::LayerAttention => "ns.layer.attention",
                    MLIROp::LayerEmbedding => "ns.layer.embedding",
                    MLIROp::LayerLayernorm => "ns.layer.layernorm",
                    MLIROp::LayerMoe => "ns.layer.moe",
                    MLIROp::Concat => "ns.concat",
                    MLIROp::Reshape => "ns.reshape",
                    MLIROp::Transpose => "ns.transpose",
                    MLIROp::Slice => "ns.slice",
                    MLIROp::Index => "ns.index",
                    MLIROp::Scatter => "ns.scatter",
                    MLIROp::Constant => "ns.constant",
                    MLIROp::FnCall => "ns.fn.call",
                    MLIROp::OptStep => "ns.opt.step",
                    MLIROp::Fused => "ns.fused",
                    _ => "ns.op",
                };
                out.push_str(op_name);
                if !instr.result_id.is_empty() {
                    out.push_str(&format!(" %{}", instr.result_id));
                }
                for op in &instr.operands {
                    out.push_str(&format!(" %{}", op));
                }
                if !instr.result_type.dims.is_empty() {
                    out.push_str(&format!(" : {}", tensor_type_to_string(&instr.result_type)));
                }
                out.push_str(&format!(" // {}\n", instr.comment));
            }
            if !fn_.return_id.is_empty() {
                out.push_str(&format!("  ns.return %{}\n", fn_.return_id));
            }
            out.push_str("}\n");
        }
        out
    }

    // -------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------

    fn collect_aliases(&mut self, program: &mut Program) {
        for stmt in program.top_level.iter() {
            if stmt.kind == StmtKind::TypeDecl {
                if let Some(ref alias_expr) = stmt.alias_expr {
                    match alias_expr.kind {
                        ExprKind::LiteralInt => {
                            if let Ok(v) = alias_expr.token.value.parse::<i64>() {
                                self.aliases.insert(stmt.alias_name.clone(), v);
                            }
                        }
                        ExprKind::Identifier => {
                            let mut v = 0i64;
                            if self.resolve_dim_int(alias_expr, &mut v) {
                                self.aliases.insert(stmt.alias_name.clone(), v);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    fn compile_fn(&mut self, stmt: &Stmt, module: &mut MLIRModule) {
        self.reset_locals();
        let mut fn_ = MLIRFunction::default();
        fn_.name = stmt.fn_name.clone();
        if let Some(ref body) = stmt.body {
            self.compile_stmt(body, &mut fn_);
        }
        module.functions.push(fn_);
    }

    fn compile_network(&mut self, stmt: &Stmt, module: &mut MLIRModule) {
        self.reset_locals();
        self.layer_meta.clear();
        self.layer_weight_id.clear();

        let mut fn_ = MLIRFunction::default();
        fn_.name = stmt.network_name.clone();

        // Weight buffer allocs for trainable layers, shared by forward() AND
        // the train() method.
        let mut weight_allocs: Vec<MLIRInstr> = Vec::new();

        for layer in stmt.layers.iter() {
            let mut meta = LayerMeta::default();
            meta.type_name = layer.layer_type.clone();

            for p in layer.layer_params.iter() {
                if p.name == "activation" {
                    if let Some(ref v) = p.value {
                        let av = v.token.value.clone();
                        if !av.is_empty() && av != "Identity" {
                            meta.activation = av;
                        }
                    }
                } else if p.name == "rate" {
                    if let Some(ref v) = p.value {
                        if let Ok(rv) = v.token.value.parse::<f64>() {
                            meta.dropout_rate = rv;
                        }
                    }
                } else if p.name == "heads" || p.name == "num_heads" {
                    if let Some(ref v) = p.value {
                        let mut hv = 0i64;
                        if self.resolve_dim_int(v, &mut hv) {
                            meta.num_heads = hv;
                        }
                    }
                } else if p.name == "causal" {
                    if let Some(ref v) = p.value {
                        meta.causal = v.token.value == "true" || v.token.value == "1" || v.token.value == "yes";
                    }
                } else if p.name == "experts" || p.name == "num_experts" {
                    if let Some(ref v) = p.value {
                        let mut ev = 0i64;
                        if self.resolve_dim_int(v, &mut ev) {
                            meta.num_experts = ev;
                        }
                    }
                } else if p.name == "ffn_dim" || p.name == "ffn" || p.name == "hidden"
                    || p.name == "d_ff" || p.name == "intermediate"
                {
                    if let Some(ref v) = p.value {
                        let mut fv = 0i64;
                        if self.resolve_dim_int(v, &mut fv) {
                            meta.ffn_dim = fv;
                        }
                    }
                } else if p.name == "initial_experts" || p.name == "active"
                    || p.name == "k" || p.name == "initial_active"
                {
                    if let Some(ref v) = p.value {
                        let mut kv = 0i64;
                        if self.resolve_dim_int(v, &mut kv) {
                            meta.initial_experts = kv;
                        }
                    }
                } else if p.name == "d_model" || p.name == "n_embd" || p.name == "dim" {
                    if let Some(ref v) = p.value {
                        let mut dv = 0i64;
                        if self.resolve_dim_int(v, &mut dv) {
                            meta.emb_dim = dv;
                        }
                    }
                } else if p.name == "vocab" || p.name == "vocab_size" {
                    if let Some(ref v) = p.value {
                        let mut vv = 0i64;
                        if self.resolve_dim_int(v, &mut vv) {
                            meta.vocab_size = vv;
                        }
                    }
                }
            }

            let trainable = meta.type_name == "Dense" || meta.type_name == "Linear"
                || meta.type_name == "Embedding"
                || meta.type_name == "Attention"
                || meta.type_name == "MultiHeadAttention"
                || meta.type_name == "MoE"
                || meta.type_name == "MixtureOfExperts";

            if trainable {
                if meta.type_name == "Dense" || meta.type_name == "Linear" {
                    let mut in_d: i64 = -1;
                    let mut out_d: i64 = -1;
                    for p in layer.layer_params.iter() {
                        if p.name == "in" {
                            if let Some(ref v) = p.value {
                                let mut vi = 0i64;
                                if self.resolve_dim_int(v, &mut vi) {
                                    in_d = vi;
                                }
                            }
                        } else if p.name == "out" {
                            if let Some(ref v) = p.value {
                                let mut vo = 0i64;
                                if self.resolve_dim_int(v, &mut vo) {
                                    out_d = vo;
                                }
                            }
                        }
                    }
                    if in_d > 0 && out_d > 0 {
                        let weight_name = format!("{}_w", layer.layer_name);
                        let mut instr = MLIRInstr::new(MLIROp::TensorAlloc, weight_name.clone());
                        instr.result_type = TensorType::new(
                            vec![DimExpr::constant(in_d), DimExpr::constant(out_d)],
                            Dtype::Float32,
                        );
                        instr.comment = format!("allocate weight [{}, {}]", in_d, out_d);
                        self.layer_weight_id
                            .insert(layer.layer_name.clone(), weight_name.clone());
                        meta.weight_ids.push(weight_name);
                        weight_allocs.push(instr);
                    }
                } else if meta.type_name == "Embedding" {
                    let v = meta.vocab_size;
                    let d = meta.emb_dim;
                    if v > 0 && d > 0 {
                        let weight_name = format!("{}_w", layer.layer_name);
                        let mut instr = MLIRInstr::new(MLIROp::TensorAlloc, weight_name.clone());
                        instr.result_type = TensorType::new(
                            vec![DimExpr::constant(v), DimExpr::constant(d)],
                            Dtype::Float32,
                        );
                        instr.comment = format!("allocate embedding [{}, {}]", v, d);
                        self.layer_weight_id
                            .insert(layer.layer_name.clone(), weight_name.clone());
                        meta.weight_ids.push(weight_name);
                        weight_allocs.push(instr);
                    }
                } else if meta.type_name == "Attention" || meta.type_name == "MultiHeadAttention" {
                    let d = meta.emb_dim;
                    let h = meta.num_heads;
                    if d > 0 && h > 0 && d % h == 0 {
                        for suffix in &["_q", "_k", "_v", "_o"] {
                            let weight_name = format!("{}{}_w", layer.layer_name, suffix);
                            let mut instr =
                                MLIRInstr::new(MLIROp::TensorAlloc, weight_name.clone());
                            instr.result_type = TensorType::new(
                                vec![DimExpr::constant(d), DimExpr::constant(d)],
                                Dtype::Float32,
                            );
                            instr.comment = format!("allocate attention weight [{}, {}]", d, d);
                            meta.weight_ids.push(weight_name);
                            weight_allocs.push(instr);
                        }
                    }
                }
            }

            if meta.type_name == "MoE" || meta.type_name == "MixtureOfExperts" {
                let d = meta.emb_dim;
                let e = meta.num_experts;
                if d > 0 && e > 0 {
                    let h = if meta.ffn_dim > 0 {
                        meta.ffn_dim
                    } else {
                        4 * d
                    };
                    meta.ffn_dim = h;
                    if meta.initial_experts <= 0 || meta.initial_experts > e {
                        meta.initial_experts = e;
                    }

                    let gate_name = format!("{}_g_w", layer.layer_name);
                    let e1_name = format!("{}_e1_w", layer.layer_name);
                    let e2_name = format!("{}_e2_w", layer.layer_name);

                    let mut gate =
                        MLIRInstr::new(MLIROp::TensorAlloc, gate_name.clone());
                    gate.result_type = TensorType::new(
                        vec![DimExpr::constant(d), DimExpr::constant(e)],
                        Dtype::Float32,
                    );
                    gate.comment = format!("allocate MoE router [{}, {}]", d, e);

                    let mut e1 =
                        MLIRInstr::new(MLIROp::TensorAlloc, e1_name.clone());
                    e1.result_type = TensorType::new(
                        vec![DimExpr::constant(e), DimExpr::constant(d * h)],
                        Dtype::Float32,
                    );
                    e1.comment = format!(
                        "allocate MoE expert FFN-1 [{}, {} -> {}]",
                        e, d, h
                    );

                    let mut e2 =
                        MLIRInstr::new(MLIROp::TensorAlloc, e2_name.clone());
                    e2.result_type = TensorType::new(
                        vec![DimExpr::constant(e), DimExpr::constant(h * d)],
                        Dtype::Float32,
                    );
                    e2.comment = format!(
                        "allocate MoE expert FFN-2 [{}, {} -> {}]",
                        e, h, d
                    );

                    meta.weight_ids.push(gate_name);
                    meta.weight_ids.push(e1_name);
                    meta.weight_ids.push(e2_name);
                    weight_allocs.push(gate);
                    weight_allocs.push(e1);
                    weight_allocs.push(e2);
                }
            }

            self.layer_meta
                .insert(layer.layer_name.clone(), meta);
        }

        // Forward pass(es) -> inference function.
        for w in weight_allocs.iter() {
            fn_.instructions.push(w.clone());
        }
        for method in stmt.methods.iter() {
            if method.kind == StmtKind::ForwardDecl {
                if let Some(ref body) = method.body {
                    self.compile_stmt(body, &mut fn_);
                }
            }
        }
        module.functions.push(fn_);
        module.functions.last_mut().unwrap().is_train = false;

        // train() method(s) -> separate function sharing the SAME weight allocs.
        let mut tfn = MLIRFunction::default();
        tfn.name = format!("{}_train", stmt.network_name);
        tfn.is_train = true;
        let mut have_train = false;
        for w in weight_allocs.iter() {
            tfn.instructions.push(w.clone());
        }
        for method in stmt.methods.iter() {
            if method.kind == StmtKind::TrainDecl {
                if let Some(ref body) = method.body {
                    self.compile_stmt(body, &mut tfn);
                    have_train = true;
                }
            }
        }
        if have_train {
            module.functions.push(tfn);
        }
    }

    fn compile_stmt(&mut self, stmt: &Stmt, fn_: &mut MLIRFunction) {
        match stmt.kind {
            StmtKind::ExprStmt => {
                if let Some(ref expr) = stmt.expr {
                    self.compile_expr(expr, fn_);
                }
            }
            StmtKind::ReturnStmt => {
                if let Some(ref init_expr) = stmt.init_expr {
                    let out = self.compile_expr_value(init_expr, fn_);
                    fn_.return_id = out;
                }
            }
            StmtKind::VarAssign => {
                if let Some(ref init_expr) = stmt.init_expr {
                    let out = self.compile_expr_value(init_expr, fn_);
                    if !stmt.var_name.is_empty() && !out.is_empty() {
                        self.local_var_ids.insert(stmt.var_name.clone(), out);
                    }
                }
            }
            StmtKind::Block => {
                for s in stmt.statements.iter() {
                    self.compile_stmt(s, fn_);
                }
            }
            StmtKind::GradBlock => {
                let mut marker = MLIRInstr::new(MLIROp::Grad, String::new());
                marker.comment = "gradient block (forward)".to_string();
                fn_.instructions.push(marker);

                let fwd_start = fn_.instructions.len();
                if let Some(ref grad_body) = stmt.grad_body {
                    self.compile_stmt(grad_body, fn_);
                }
                let fwd_end = fn_.instructions.len();

                // Reverse-mode lowering over the range [fwd_start, fwd_end).
                let mut g: HashMap<String, String> = HashMap::new();
                let mut i = fwd_end;
                while i > fwd_start {
                    // Copy by value: backward instrs are appended to the same
                    // vector.
                    let orig = fn_.instructions[i - 1].clone();
                    match orig.op {
                        MLIROp::CrossEntropy => {
                            let mut lg = MLIRInstr::new(
                                MLIROp::LossGrad,
                                self.new_temp("g"),
                            );
                            lg.operands = orig.operands.clone();
                            lg.attribute = "cross_entropy".to_string();
                            lg.comment = "d(cross_entropy)/d(preds)".to_string();
                            fn_.instructions.push(lg.clone());
                            self.set_grad(&mut g, &orig.result_id, &lg.result_id, fn_);
                            if let Some(first) = orig.operands.first() {
                                self.set_grad(&mut g, first, &lg.result_id, fn_);
                            }
                        }
                        MLIROp::ElementwiseBinop => {
                            if let Some(cit) = g.get(&orig.result_id).cloned() {
                                if orig.operands.len() >= 2 {
                                    let mut bgl = MLIRInstr::new(
                                        MLIROp::BinopGrad,
                                        self.new_temp("g"),
                                    );
                                    bgl.operands = vec![
                                        cit.clone(),
                                        orig.operands[0].clone(),
                                        orig.operands[1].clone(),
                                    ];
                                    bgl.attribute = orig.attribute.clone();
                                    bgl.int_attr = 0;
                                    bgl.comment = format!(
                                        "d{} = binop_grad(d{}, {}, {}, {}, lhs)",
                                        orig.operands[0],
                                        orig.result_id,
                                        orig.operands[0],
                                        orig.operands[1],
                                        orig.attribute
                                    );
                                    fn_.instructions.push(bgl.clone());
                                    self.set_grad(&mut g, &orig.operands[0], &bgl.result_id, fn_);

                                    let mut bgr = MLIRInstr::new(
                                        MLIROp::BinopGrad,
                                        self.new_temp("g"),
                                    );
                                    bgr.operands = vec![
                                        cit,
                                        orig.operands[0].clone(),
                                        orig.operands[1].clone(),
                                    ];
                                    bgr.attribute = orig.attribute.clone();
                                    bgr.int_attr = 1;
                                    bgr.comment = format!(
                                        "d{} = binop_grad(d{}, {}, {}, {}, rhs)",
                                        orig.operands[1],
                                        orig.result_id,
                                        orig.operands[0],
                                        orig.operands[1],
                                        orig.attribute
                                    );
                                    fn_.instructions.push(bgr.clone());
                                    self.set_grad(&mut g, &orig.operands[1], &bgr.result_id, fn_);
                                }
                            }
                        }
                        MLIROp::Matmul => {
                            if let Some(cit) = g.get(&orig.result_id).cloned() {
                                if orig.operands.len() >= 2 {
                                    let mut ia = MLIRInstr::new(
                                        MLIROp::MatmulGradA,
                                        self.new_temp("g"),
                                    );
                                    ia.operands = vec![
                                        cit.clone(),
                                        orig.operands[1].clone(),
                                    ];
                                    ia.comment = format!(
                                        "d{} = d{} @ {}^T",
                                        orig.operands[0], orig.result_id, orig.operands[1]
                                    );
                                    fn_.instructions.push(ia.clone());
                                    self.set_grad(&mut g, &orig.operands[0], &ia.result_id, fn_);

                                    let mut iw = MLIRInstr::new(
                                        MLIROp::MatmulGradW,
                                        self.new_temp("g"),
                                    );
                                    iw.operands = vec![
                                        orig.operands[0].clone(),
                                        cit,
                                        orig.operands[1].clone(),
                                    ];
                                    iw.comment = format!(
                                        "d{} = {}^T @ d{}",
                                        orig.operands[1], orig.operands[0], orig.result_id
                                    );
                                    fn_.instructions.push(iw.clone());
                                    self.set_grad(&mut g, &orig.operands[1], &iw.result_id, fn_);
                                }
                            }
                        }
                        MLIROp::Layernorm | MLIROp::LayerLayernorm => {
                            if let Some(cit) = g.get(&orig.result_id).cloned() {
                                if !orig.operands.is_empty() {
                                    let mut lg = MLIRInstr::new(
                                        MLIROp::LayernormGrad,
                                        self.new_temp("g"),
                                    );
                                    lg.operands = vec![
                                        cit,
                                        orig.operands[0].clone(),
                                    ];
                                    lg.attribute = Self::activation_name_from_op(orig.op).to_string();
                                    lg.comment = format!(
                                        "d{} = layernorm_grad(d{}, {})",
                                        orig.operands[0], orig.result_id, orig.operands[0]
                                    );
                                    fn_.instructions.push(lg.clone());
                                    self.set_grad(&mut g, &orig.operands[0], &lg.result_id, fn_);
                                }
                            }
                        }
                        MLIROp::Relu
                        | MLIROp::LeakyRelu
                        | MLIROp::Sigmoid
                        | MLIROp::Tanh
                        | MLIROp::Swish
                        | MLIROp::Gelu
                        | MLIROp::Silu
                        | MLIROp::Identity
                        | MLIROp::Softmax => {
                            if let Some(cit) = g.get(&orig.result_id).cloned() {
                                if !orig.operands.is_empty() {
                                    let mut ig = MLIRInstr::new(
                                        MLIROp::ActivationGrad,
                                        self.new_temp("g"),
                                    );
                                    ig.operands = vec![
                                        cit,
                                        orig.operands[0].clone(),
                                    ];
                                    ig.attribute = Self::activation_name_from_op(orig.op).to_string();
                                    ig.comment = format!(
                                        "d{} = d{} * act'(in)",
                                        orig.operands[0], orig.result_id
                                    );
                                    fn_.instructions.push(ig.clone());
                                    self.set_grad(&mut g, &orig.operands[0], &ig.result_id, fn_);
                                }
                            }
                        }
                        MLIROp::Dropout => {
                            if !orig.operands.is_empty() {
                                if let Some(cit) = g.get(&orig.result_id).cloned() {
                                    let mut ig = MLIRInstr::new(
                                        MLIROp::ActivationGrad,
                                        self.new_temp("g"),
                                    );
                                    ig.operands = vec![
                                        cit,
                                        orig.result_id.clone(),
                                    ];
                                    ig.attribute = "dropout".to_string();
                                    ig.comment = format!(
                                        "d{} = d{} * dropout_mask",
                                        orig.operands[0], orig.result_id
                                    );
                                    fn_.instructions.push(ig.clone());
                                    self.set_grad(&mut g, &orig.operands[0], &ig.result_id, fn_);
                                }
                            }
                        }
                        MLIROp::LayerEmbedding => {
                            if let Some(cit) = g.get(&orig.result_id).cloned() {
                                if orig.operands.len() >= 2 {
                                    let mut wg = MLIRInstr::new(
                                        MLIROp::EmbeddingGradW,
                                        self.new_temp("g"),
                                    );
                                    wg.operands = vec![
                                        cit,
                                        orig.operands[1].clone(),
                                        orig.operands[0].clone(),
                                    ];
                                    wg.comment = format!(
                                        "d{} = scatter_add(d{})",
                                        orig.operands[0], orig.result_id
                                    );
                                    fn_.instructions.push(wg.clone());
                                    self.set_grad(&mut g, &orig.operands[0], &wg.result_id, fn_);
                                }
                            }
                        }
                        MLIROp::LayerMoe => {
                            if let Some(cit) = g.get(&orig.result_id).cloned() {
                                if orig.operands.len() >= 4 {
                                    let base = orig.operands.clone();
                                    // dx (gate path + routed expert path)
                                    let mut mx = MLIRInstr::new(
                                        MLIROp::MoeGradX,
                                        self.new_temp("g"),
                                    );
                                    let mut ops = Vec::new();
                                    ops.push(cit.clone());
                                    ops.extend(base.clone());
                                    mx.operands = ops;
                                    mx.attribute = orig.attribute.clone();
                                    mx.int_attr = orig.int_attr;
                                    mx.float_attr = orig.float_attr;
                                    mx.ints_attr = orig.ints_attr.clone();
                                    mx.comment = "dx(moe)".to_string();
                                    fn_.instructions.push(mx.clone());
                                    self.set_grad(&mut g, &base[0], &mx.result_id, fn_);
                                    // dWg (router weights)
                                    let mut mg = MLIRInstr::new(
                                        MLIROp::MoeGradWg,
                                        self.new_temp("g"),
                                    );
                                    mg.operands = mx.operands.clone();
                                    mg.attribute = orig.attribute.clone();
                                    mg.int_attr = orig.int_attr;
                                    mg.float_attr = orig.float_attr;
                                    mg.ints_attr = orig.ints_attr.clone();
                                    mg.comment = "dWg(moe)".to_string();
                                    fn_.instructions.push(mg.clone());
                                    self.set_grad(&mut g, &base[1], &mg.result_id, fn_);
                                    // dWe1 / dWe2
                                    let mut me1 = MLIRInstr::new(
                                        MLIROp::MoeGradWe1,
                                        self.new_temp("g"),
                                    );
                                    me1.operands = mx.operands.clone();
                                    me1.attribute = orig.attribute.clone();
                                    me1.int_attr = orig.int_attr;
                                    me1.float_attr = orig.float_attr;
                                    me1.ints_attr = orig.ints_attr.clone();
                                    me1.comment = "dWe1(moe)".to_string();
                                    fn_.instructions.push(me1.clone());
                                    self.set_grad(&mut g, &base[2], &me1.result_id, fn_);
                                    let mut me2 = MLIRInstr::new(
                                        MLIROp::MoeGradWe2,
                                        self.new_temp("g"),
                                    );
                                    me2.operands = mx.operands.clone();
                                    me2.attribute = orig.attribute.clone();
                                    me2.int_attr = orig.int_attr;
                                    me2.float_attr = orig.float_attr;
                                    me2.ints_attr = orig.ints_attr.clone();
                                    me2.comment = "dWe2(moe)".to_string();
                                    fn_.instructions.push(me2.clone());
                                    self.set_grad(&mut g, &base[3], &me2.result_id, fn_);
                                }
                                }
                        }
                        MLIROp::LayerAttention => {
                            if let Some(cit2) = g.get(&orig.result_id).cloned() {
                                if orig.operands.len() >= 5 {
                                    let h_str = if orig.attribute.is_empty() {
                                        "1".to_string()
                                    } else {
                                        orig.attribute.clone()
                                    };
                                    let _d_str = orig.int_attr.to_string();
                                    let mut ax = MLIRInstr::new(
                                        MLIROp::AttentionGradX,
                                        self.new_temp("g"),
                                    );
                                    ax.operands = vec![
                                        cit2,
                                        orig.operands[0].clone(),
                                        orig.operands[1].clone(),
                                        orig.operands[2].clone(),
                                        orig.operands[3].clone(),
                                        orig.operands[4].clone(),
                                    ];
                                    ax.attribute = h_str.clone();
                                    ax.int_attr = orig.int_attr;
                                    ax.ints_attr = orig.ints_attr.clone();
                                    ax.comment = "dx(attention)".to_string();
                                    fn_.instructions.push(ax.clone());
                                    self.set_grad(&mut g, &orig.operands[0], &ax.result_id, fn_);

                                    let wops = [
                                        MLIROp::AttentionGradWq,
                                        MLIROp::AttentionGradWk,
                                        MLIROp::AttentionGradWv,
                                        MLIROp::AttentionGradWo,
                                    ];
                                    for (qi, wop) in wops.iter().enumerate() {
                                        let mut aq = MLIRInstr::new(
                                            *wop,
                                            self.new_temp("g"),
                                        );
                                        aq.operands = ax.operands.clone();
                                        aq.attribute = h_str.clone();
                                        aq.int_attr = orig.int_attr;
                                        aq.ints_attr = orig.ints_attr.clone();
                                        fn_.instructions.push(aq.clone());
                                        self.set_grad(&mut g, &orig.operands[1 + qi], &aq.result_id, fn_);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                    i -= 1;
                }

                // Optimizer step: pair each weight alloc with its gradient.
                let mut opt_ops: Vec<String> = Vec::new();
                for instr in fn_.instructions.iter() {
                    if instr.op != MLIROp::TensorAlloc {
                        continue;
                    }
                    if let Some(gradient) = g.get(&instr.result_id) {
                        opt_ops.push(instr.result_id.clone());
                        opt_ops.push(gradient.clone());
                    }
                }
                if !opt_ops.is_empty() {
                    let mut opt = MLIRInstr::new(MLIROp::OptStep, String::new());
                    opt.operands = opt_ops;
                    opt.attribute = "muon".to_string();
                    opt.comment = "optimizer step (Muon / AdamW)".to_string();
                    fn_.instructions.push(opt);
                }
            }
            StmtKind::TrainDecl => {
                if let Some(ref body) = stmt.body {
                    self.compile_stmt(body, fn_);
                }
            }
            StmtKind::ForwardDecl => {
                let mut instr = MLIRInstr::new(MLIROp::Forward, String::new());
                instr.comment = "forward() entry".to_string();
                fn_.instructions.push(instr);
                if let Some(ref body) = stmt.body {
                    self.compile_stmt(body, fn_);
                }
            }
            StmtKind::LayerDecl => {
                let mut op = MLIROp::LayerDense;
                let type_name = stmt.layer_type.clone();
                if type_name == "Dense" || type_name == "Linear" {
                    op = MLIROp::LayerDense;
                } else if type_name == "Dropout" {
                    op = MLIROp::LayerDropout;
                } else if type_name == "Attention" {
                    op = MLIROp::LayerAttention;
                } else if type_name == "Embedding" {
                    op = MLIROp::LayerEmbedding;
                } else if type_name == "LayerNorm" {
                    op = MLIROp::LayerLayernorm;
                }
                let mut instr = MLIRInstr::new(op, stmt.layer_name.clone());
                instr.comment = format!("layer {}", type_name);
                // Extract params into attr string, e.g. "in:128,out:512"
                let mut params = String::new();
                for (i, p) in stmt.layer_params.iter().enumerate() {
                    if i > 0 {
                        params.push(',');
                    }
                    params.push_str(&p.name);
                    if let Some(ref v) = p.value {
                        params.push(':');
                        params.push_str(&v.token.value);
                    }
                }
                instr.attribute = params;
                fn_.instructions.push(instr);
            }
            StmtKind::VarDecl => {
                if let Some(ref var_type) = stmt.var_type {
                    if var_type.is_tensor() {
                        let mut instr = MLIRInstr::new(
                            MLIROp::TensorAlloc,
                            stmt.var_name.clone(),
                        );
                        instr.result_type = var_type.tensor_type.clone();
                        instr.comment = format!(
                            "allocate {}",
                            tensor_type_to_string(&var_type.tensor_type)
                        );
                        fn_.instructions.push(instr);
                    }
                }
                if let Some(ref init_expr) = stmt.init_expr {
                    let out = self.compile_expr_value(init_expr, fn_);
                    if !stmt.var_name.is_empty() && !out.is_empty() {
                        self.local_var_ids.insert(stmt.var_name.clone(), out);
                    }
                }
            }
            _ => {}
        }
    }

    fn compile_expr_value(&mut self, expr: &Expr, fn_: &mut MLIRFunction) -> String {
        let out = self.compile_expr(expr, fn_);
        out
    }

    /// Compile an expression, returning the value id. Also emits instructions.
    fn compile_expr(&mut self, expr: &Expr, fn_: &mut MLIRFunction) -> String {
        match expr.kind {
            ExprKind::LiteralInt | ExprKind::LiteralFloat | ExprKind::LiteralBool => {
                let instr_id = self.new_temp("c");
                let mut instr = MLIRInstr::new(MLIROp::Constant, instr_id.clone());
                instr.attribute = expr.token.value.clone();
                instr.comment = format!("constant {}", expr.token.value);
                fn_.instructions.push(instr);
                instr_id
            }
            ExprKind::Identifier => {
                let mut out_id = expr.token.value.clone();
                // Local var binding first.
                if let Some(lit) = self.local_var_ids.get(&out_id) {
                    out_id = lit.clone();
                } else if let Some(wit) = self.layer_weight_id.get(&out_id) {
                    out_id = wit.clone();
                }
                // Note: inferred_type is set by shape checker; here we skip type propagation
                // as the Rust AST doesn't attach it in the same way.
                out_id
            }
            ExprKind::MatmulOp => {
                self.lower_matmul(expr, fn_)
            }
            ExprKind::PipelineOp => {
                self.lower_pipeline(expr, fn_)
            }
            ExprKind::BinaryOp => {
                let lhs = if let Some(ref left) = expr.left {
                    self.compile_expr(left, fn_)
                } else {
                    String::new()
                };
                let rhs = if let Some(ref right) = expr.right {
                    self.compile_expr(right, fn_)
                } else {
                    String::new()
                };
                let instr_id = self.new_temp("e");
                let mut instr = MLIRInstr::new(MLIROp::ElementwiseBinop, instr_id.clone());
                instr.operands = vec![lhs, rhs];
                instr.attribute = expr.token.value.clone();
                fn_.instructions.push(instr);
                instr_id
            }
            ExprKind::FunctionCall => {
                let func = if let Some(ref op) = expr.operand {
                    op.token.value.clone()
                } else {
                    expr.callee.clone()
                };
                self.lower_function_call(&func, expr, fn_)
            }
            ExprKind::UnaryOp => {
                if let Some(ref op) = expr.operand {
                    self.compile_expr(op, fn_)
                } else {
                    String::new()
                }
            }
            ExprKind::IndexOp => {
                if let Some(ref op) = expr.operand {
                    self.compile_expr(op, fn_)
                } else {
                    String::new()
                }
            }
            _ => String::new(),
        }
    }

    fn lower_matmul(&mut self, expr: &Expr, fn_: &mut MLIRFunction) -> String {
        let lhs = if let Some(ref left) = expr.left {
            self.compile_expr(left, fn_)
        } else {
            String::new()
        };
        let rhs = if let Some(ref right) = expr.right {
            self.compile_expr(right, fn_)
        } else {
            String::new()
        };
        let instr_id = self.new_temp("mm");
        let mut instr = MLIRInstr::new(MLIROp::Matmul, instr_id.clone());
        instr.operands = vec![lhs.clone(), rhs.clone()];
        if let Some(ref t) = expr.inferred_type {
            if t.is_tensor() {
                instr.result_type = t.tensor_type.clone();
            }
        }
        instr.comment = format!("{} @ {}", lhs, rhs);
        fn_.instructions.push(instr);
        instr_id
    }

    fn lower_pipeline(&mut self, expr: &Expr, fn_: &mut MLIRFunction) -> String {
        let input = if let Some(ref left) = expr.left {
            self.compile_expr(left, fn_)
        } else {
            String::new()
        };
        self.lower_pipeline_apply(expr.right.as_deref(), fn_, input)
    }

    fn lower_pipeline_apply(
        &mut self,
        expr: Option<&Expr>,
        fn_: &mut MLIRFunction,
        input: String,
    ) -> String {
        let expr = match expr {
            Some(e) => e,
            None => return input,
        };
        match expr.kind {
            ExprKind::PipelineOp => {
                let stage_out = self.lower_pipeline_apply(expr.left.as_deref(), fn_, input);
                if let Some(ref right) = expr.right {
                    if right.kind == ExprKind::PipelineOp {
                        self.lower_pipeline_apply(Some(right), fn_, stage_out)
                    } else {
                        self.apply_pipeline_stage(&right.token.value, stage_out, fn_)
                    }
                } else {
                    stage_out
                }
            }
            _ => self.apply_pipeline_stage(&expr.token.value, input, fn_),
        }
    }

    fn lower_activation(&mut self, act: &str, input: &str, fn_: &mut MLIRFunction) -> String {
        let op = Self::activation_op_from_name(act);
        let instr_id = self.new_temp("act");
        let mut instr = MLIRInstr::new(op, instr_id.clone());
        instr.operands = vec![input.to_string()];
        instr.comment = format!("{}({})", act, input);
        fn_.instructions.push(instr);
        instr_id
    }

    fn apply_pipeline_stage(
        &mut self,
        wname: &str,
        input: String,
        fn_: &mut MLIRFunction,
    ) -> String {
        let meta = self.layer_meta.get(wname).cloned();
        if let Some(ref m) = meta {
            match m.type_name.as_str() {
                "Dropout" => {
                    let instr_id = self.new_temp("fc");
                    let mut instr = MLIRInstr::new(MLIROp::Dropout, instr_id.clone());
                    instr.operands = vec![input.clone()];
                    instr.float_attr = m.dropout_rate;
                    instr.comment = format!("dropout({}, p={})", input, crate::nns::fmt_float(m.dropout_rate));
                    fn_.instructions.push(instr);
                    return instr_id;
                }
                "Embedding" => {
                    let wid = self
                        .layer_weight_id
                        .get(wname)
                        .cloned()
                        .unwrap_or_else(|| wname.to_string());
                    let instr_id = self.new_temp("emb");
                    let mut instr =
                        MLIRInstr::new(MLIROp::LayerEmbedding, instr_id.clone());
                    instr.operands = vec![wid.clone(), input.clone()];
                    instr.result_type = TensorType::new(
                        vec![
                            DimExpr::dynamic(),
                            DimExpr::constant(m.emb_dim),
                        ],
                        Dtype::Float32,
                    );
                    instr.comment = format!("embedding({}, {})", input, wid);
                    fn_.instructions.push(instr);
                    return instr_id;
                }
                "LayerNorm" | "Normalize" => {
                    let instr_id = self.new_temp("ln");
                    let mut instr = MLIRInstr::new(MLIROp::Layernorm, instr_id.clone());
                    instr.operands = vec![input.clone()];
                    // Mirror C++ mlir_compiler.cpp: instr.result_type = in.type
                    let mut found: Option<TensorType> = None;
                    for prev in fn_.instructions.iter().rev() {
                        if prev.result_id == input && !prev.result_type.dims.is_empty() {
                            found = Some(prev.result_type.clone());
                            break;
                        }
                    }
                    if let Some(rt) = found {
                        instr.result_type = rt;
                    } else {
                        // Fallback when in.type is empty (e.g. forward param with no
                        // materialized type). Use Symbolic("T") for byte-identical dump
                        // (C++ prints S/T, not Dynamic) and try to infer feature width
                        // from the most recent typed 2-D value so codegen gets 512 not 1.
                        let mut fallback: Option<TensorType> = None;
                        for prev in fn_.instructions.iter().rev() {
                            if prev.result_type.dims.len() >= 2 {
                                fallback = Some(prev.result_type.clone());
                                break;
                            }
                        }
                        if let Some(fb) = fallback {
                            instr.result_type = fb;
                        } else {
                            instr.result_type = TensorType::new(
                                vec![DimExpr::symbolic("T"), DimExpr::dynamic()],
                                Dtype::Float32,
                            );
                        }
                    }
                    instr.comment = format!("layernorm({})", input);
                    fn_.instructions.push(instr);
                    return instr_id;
                }
                "Attention" | "MultiHeadAttention" => {
                    if m.weight_ids.len() < 4 {
                        eprintln!(
                            "Attention layer '{}' has no materialized q/k/v/o weights",
                            wname
                        );
                        return input;
                    }
                    let instr_id = self.new_temp("attn");
                    let mut instr =
                        MLIRInstr::new(MLIROp::LayerAttention, instr_id.clone());
                    instr.operands = vec![
                        input.clone(),
                        m.weight_ids[0].clone(),
                        m.weight_ids[1].clone(),
                        m.weight_ids[2].clone(),
                        m.weight_ids[3].clone(),
                    ];
                    instr.attribute = m.num_heads.to_string();
                    instr.int_attr = m.emb_dim;
                    instr.ints_attr = vec![if m.causal { 1 } else { 0 }];
                    instr.result_type = TensorType::new(
                        vec![
                            DimExpr::dynamic(),
                            DimExpr::constant(m.emb_dim),
                        ],
                        Dtype::Float32,
                    );
                    instr.comment = format!(
                        "attention({}, heads={})",
                        input, m.num_heads
                    );
                    fn_.instructions.push(instr);
                    return instr_id;
                }
                "MoE" | "MixtureOfExperts" => {
                    if m.weight_ids.len() < 3 {
                        eprintln!(
                            "MoE layer '{}' has no materialized router/FFN weights",
                            wname
                        );
                        return input;
                    }
                    let instr_id = self.new_temp("moe");
                    let mut instr =
                        MLIRInstr::new(MLIROp::LayerMoe, instr_id.clone());
                    instr.operands = vec![
                        input.clone(),
                        m.weight_ids[0].clone(),
                        m.weight_ids[1].clone(),
                        m.weight_ids[2].clone(),
                    ];
                    instr.attribute = m.num_experts.to_string();
                    instr.int_attr = m.emb_dim;
                    instr.float_attr = m.ffn_dim as f64;
                    instr.ints_attr = vec![m.initial_experts];
                    instr.result_type = TensorType::new(
                        vec![
                            DimExpr::dynamic(),
                            DimExpr::constant(m.emb_dim),
                        ],
                        Dtype::Float32,
                    );
                    instr.comment = format!(
                        "moe({}, experts={}, ffn={})",
                        input, m.num_experts, m.ffn_dim
                    );
                    fn_.instructions.push(instr);
                    return instr_id;
                }
                "Dense" | "Linear" => {}
                _ => {}
            }
        }

        let wid = self
            .layer_weight_id
            .get(wname)
            .cloned()
            .unwrap_or_else(|| wname.to_string());
        let instr_id = self.new_temp("fc");
        let mut instr = MLIRInstr::new(MLIROp::Matmul, instr_id.clone());
        instr.operands = vec![input.clone(), wid.clone()];
        instr.comment = format!("{} -> {}", input, wname);
        fn_.instructions.push(instr);

        let act = meta.as_ref().map(|m| m.activation.clone()).unwrap_or_default();
        if !act.is_empty() {
            self.lower_activation(&act, &instr_id, fn_)
        } else {
            instr_id
        }
    }

    fn lower_function_call(
        &mut self,
        func: &str,
        expr: &Expr,
        fn_: &mut MLIRFunction,
    ) -> String {
        let op = match func {
            "cross_entropy" => MLIROp::CrossEntropy,
            "Dense" | "Linear" => MLIROp::LayerDense,
            "Dropout" => MLIROp::LayerDropout,
            "LayerNorm" => MLIROp::LayerLayernorm,
            "relu" => MLIROp::Relu,
            "leaky_relu" => MLIROp::LeakyRelu,
            "sigmoid" => MLIROp::Sigmoid,
            "tanh" => MLIROp::Tanh,
            "silu" | "swish" => MLIROp::Swish,
            "gelu" => MLIROp::Gelu,
            "softmax" => MLIROp::Softmax,
            "identity" => MLIROp::Identity,
            "dropout" => MLIROp::Dropout,
            "concat" | "transpose" | "reshape" | "slice" | "index" | "scatter" => {
                return self.lower_data_movement(func, expr, fn_);
            }
            _ => MLIROp::FnCall,
        };

        let instr_id = self.new_temp("f");
        let mut instr = MLIRInstr::new(op, instr_id.clone());
        // Skip nested function calls (the self operand)
        for arg in expr.args.iter() {
            let a = self.compile_expr(arg, fn_);
            instr.operands.push(a);
        }
        if op == MLIROp::Dropout && expr.args.len() >= 2 {
            if let Some(ref rate_arg) = expr.args.get(1) {
                if rate_arg.kind == ExprKind::LiteralInt
                    || rate_arg.kind == ExprKind::LiteralFloat
                {
                    if let Ok(rv) = rate_arg.token.value.parse::<f64>() {
                        instr.float_attr = rv;
                    }
                }
            }
        }
        instr.comment = format!("{}()", func);
        fn_.instructions.push(instr);
        instr_id
    }

    fn lower_data_movement(
        &mut self,
        func: &str,
        expr: &Expr,
        fn_: &mut MLIRFunction,
    ) -> String {
        let lit_int = |e: &Expr, out: &mut i64| -> bool {
            match e.kind {
                ExprKind::LiteralInt => e.token.value.parse::<i64>().map(|v| *out = v).is_ok(),
                ExprKind::LiteralFloat => {
                    e.token.value.parse::<i64>().map(|v| *out = v).is_ok()
                }
                _ => false,
            }
        };

        let (dm_op, pfx) = match func {
            "concat" => (MLIROp::Concat, "cat"),
            "transpose" => (MLIROp::Transpose, "tp"),
            "reshape" => (MLIROp::Reshape, "rs"),
            "slice" => (MLIROp::Slice, "sl"),
            "index" => (MLIROp::Index, "ix"),
            _ => (MLIROp::Scatter, "sc"),
        };

        let instr_id = self.new_temp(pfx);
        let mut instr = MLIRInstr::new(dm_op, instr_id.clone());

        if func == "concat" {
            if expr.args.len() >= 2 {
                let a_id = self.compile_expr(&expr.args[0], fn_);
                let b_id = self.compile_expr(&expr.args[1], fn_);
                instr.operands = vec![a_id, b_id];
                let mut axis = 1i64;
                if expr.args.len() >= 3 {
                    lit_int(&expr.args[2], &mut axis);
                }
                instr.attribute = axis.to_string();
            }
        } else if func == "transpose" {
            if !expr.args.is_empty() {
                let a_id = self.compile_expr(&expr.args[0], fn_);
                instr.operands = vec![a_id];
            }
        } else if func == "reshape" {
            if !expr.args.is_empty() {
                let a_id = self.compile_expr(&expr.args[0], fn_);
                instr.operands = vec![a_id];
                if expr.args.len() >= 3 {
                    let mut r = 0i64;
                    let mut c = 0i64;
                    lit_int(&expr.args[1], &mut r);
                    lit_int(&expr.args[2], &mut c);
                    instr.result_type = TensorType::new(
                        vec![DimExpr::constant(r), DimExpr::constant(c)],
                        Dtype::Float32,
                    );
                    instr.attribute = format!("{}:{}", r, c);
                }
            }
        } else if func == "slice" {
            if !expr.args.is_empty() {
                let a_id = self.compile_expr(&expr.args[0], fn_);
                instr.operands = vec![a_id];
                if expr.args.len() >= 4 {
                    let mut axis = 1i64;
                    let mut s = 0i64;
                    let mut e = 0i64;
                    lit_int(&expr.args[1], &mut axis);
                    lit_int(&expr.args[2], &mut s);
                    lit_int(&expr.args[3], &mut e);
                    instr.attribute = format!("{}:{}:{}", axis, s, e);
                }
            }
        } else {
            // index / scatter
            if !expr.args.is_empty() {
                let a_id = self.compile_expr(&expr.args[0], fn_);
                let mut axis = 1i64;
                if expr.args.len() >= 2 {
                    lit_int(&expr.args[1], &mut axis);
                }
                instr.operands = vec![a_id];
                if func == "scatter" && expr.args.len() >= 3 {
                    let upd_id = self.compile_expr(&expr.args[2], fn_);
                    instr.operands.push(upd_id);
                }
                instr.int_attr = axis;
                let first_idx = if func == "scatter" { 3 } else { 2 };
                for k in first_idx..expr.args.len() {
                    let mut v = 0i64;
                    if lit_int(&expr.args[k], &mut v) {
                        instr.ints_attr.push(v);
                    }
                }
            }
        }
        instr.comment = format!("{}()", func);
        fn_.instructions.push(instr);
        instr_id
    }
}

/// Helper: format a TensorType as a compact string (e.g. "[2, 3] float32").
fn tensor_type_to_string(tt: &TensorType) -> String {
    tt.to_string()
}
