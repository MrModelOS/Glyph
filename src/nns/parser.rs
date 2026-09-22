#![allow(dead_code)]
//! Recursive-descent parser — port of `ns/parser/parser.cpp`.

use super::ast::{
    DimExpr, Dtype, Expr, ExprKind, Param, Program, Stmt, StmtKind, TensorType, TypeNode,
};
use super::token::{Token, TokenType};
use super::{NsError, NsResult};

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, current: 0 }
    }

    // ---- Token helpers ----

    fn peek(&self) -> &Token {
        if self.current >= self.tokens.len() {
            &self.tokens[self.tokens.len() - 1]
        } else {
            &self.tokens[self.current]
        }
    }

    fn peek_ahead(&self, n: usize) -> &Token {
        let idx = (self.current + n).min(self.tokens.len().saturating_sub(1));
        &self.tokens[idx]
    }

    fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }

    fn advance(&mut self) -> Token {
        if self.current < self.tokens.len() {
            self.current += 1;
        }
        self.tokens[self.current - 1].clone()
    }

    fn check(&self, type_: TokenType) -> bool {
        self.peek().type_ == type_
    }

    fn match_(&mut self, type_: TokenType) -> bool {
        if self.check(type_) {
            self.advance();
            return true;
        }
        false
    }

    fn match_one_of(&mut self, types: &[TokenType]) -> bool {
        for &t in types {
            if self.match_(t) {
                return true;
            }
        }
        false
    }

    fn expect(&mut self, type_: TokenType, msg: &str) -> NsResult<Token> {
        if self.check(type_) {
            return Ok(self.advance());
        }
        let peek = self.peek();
        Err(NsError(format!(
            "Expected {} but got {} '{}' at {}:{}: {}",
            type_.name(),
            peek.type_.name(),
            peek.value,
            peek.line,
            peek.column,
            msg
        )))
    }

    fn error(&self, token: &Token, msg: &str) -> NsError {
        NsError(format!("Parse error at {}:{}: {}", token.line, token.column, msg))
    }

    fn at_end(&self) -> bool {
        self.peek().type_ == TokenType::EofToken
    }

    // ---- Program ----

    pub fn parse_program(&mut self) -> NsResult<Program> {
        let mut program = Program::new();
        while !self.at_end() {
            if self.match_(TokenType::OpSemicolon) {
                continue;
            }
            program.add(Box::new(self.parse_statement()?));
        }
        Ok(program)
    }

    // ---- Statements ----

    fn parse_statement(&mut self) -> NsResult<Stmt> {
        let tok = self.peek().clone();

        match tok.type_ {
            TokenType::KwVar => {
                let var_tok = self.advance();
                self.parse_var_decl(var_tok)
            }
            TokenType::KwFn => {
                let fn_tok = self.advance();
                self.parse_fn_decl(fn_tok)
            }
            TokenType::KwNetwork => {
                let net_tok = self.advance();
                self.parse_network_decl(net_tok)
            }
            TokenType::KwGrad => {
                let grad_tok = self.advance();
                self.parse_grad_block(grad_tok)
            }
            TokenType::KwReturn => self.parse_return_stmt(),
            TokenType::KwIf => self.parse_if_stmt(),
            TokenType::KwWhile => self.parse_while_stmt(),
            TokenType::KwType => {
                let type_tok = self.advance();
                self.parse_type_decl(type_tok)
            }
            TokenType::KwLayer => {
                let layer_tok = self.advance();
                self.parse_layer_decl(layer_tok)
            }
            TokenType::OpLbrace => self.parse_block(),
            TokenType::KwForward => {
                // forward(x) { return x -> fc1 -> ... }
                let fwd_tok = self.advance();
                let mut stmt = Stmt::new(StmtKind::ForwardDecl, fwd_tok);
                self.expect(TokenType::OpLparen, "expected ( after forward")?;
                if !self.check(TokenType::OpRparen) {
                    loop {
                        let p = self.advance();
                        stmt.params.push(Param {
                            name: p.value,
                            type_: None,
                            is_mut: false,
                            is_ref: false,
                        });
                        if !self.match_(TokenType::OpComma) {
                            break;
                        }
                        if self.check(TokenType::OpRparen) {
                            break;
                        }
                    }
                }
                self.expect(TokenType::OpRparen, "expected ) after forward params")?;
                stmt.body = Some(Box::new(self.parse_block()?));
                Ok(stmt)
            }
            TokenType::KwTrain => {
                // train(x, y) -> loss_type { grad { ... } }
                let t = self.advance();
                self.parse_train_decl(t)
            }
            _ => self.parse_expr_stmt(),
        }
    }

    fn parse_var_decl(&mut self, tok: Token) -> NsResult<Stmt> {
        let mut stmt = Stmt::new(StmtKind::VarDecl, tok);
        let name = self.expect(TokenType::Identifier, "Expected variable name")?;
        stmt.var_name = name.value;

        if self.match_(TokenType::OpColon) {
            stmt.var_type = Some(Box::new(self.parse_type()?));
        }

        if self.match_(TokenType::OpAssign) {
            stmt.init_expr = Some(Box::new(self.parse_expression()?));
        }

        self.match_(TokenType::OpSemicolon);
        Ok(stmt)
    }

    fn parse_fn_decl(&mut self, tok: Token) -> NsResult<Stmt> {
        let mut stmt = Stmt::new(StmtKind::FnDecl, tok);
        let name = self.expect(TokenType::Identifier, "Expected function name")?;
        stmt.fn_name = name.value;

        self.expect(TokenType::OpLparen, "expected ( after function name")?;
        while !self.check(TokenType::OpRparen) {
            let mut param = Param {
                name: String::new(),
                type_: None,
                is_mut: false,
                is_ref: false,
            };
            param.is_mut = self.match_(TokenType::KwMut);
            param.is_ref = self.match_(TokenType::KwRef);
            let pname = self.expect(TokenType::Identifier, "Expected parameter name")?;
            param.name = pname.value;
            if self.match_(TokenType::OpColon) {
                param.type_ = Some(Box::new(self.parse_type()?));
            }
            stmt.params.push(param);
            if !self.match_(TokenType::OpComma) {
                break;
            }
        }
        self.expect(TokenType::OpRparen, "expected ) after params")?;

        // Return type: "-> Type" or ": Type"
        if self.match_(TokenType::OpPipeline) || self.match_(TokenType::OpColon) {
            stmt.return_type = Some(Box::new(self.parse_type()?));
        }

        stmt.body = Some(Box::new(self.parse_block()?));
        Ok(stmt)
    }

    fn parse_train_decl(&mut self, tok: Token) -> NsResult<Stmt> {
        // train(x: Tensor[Batch, F], y: Tensor[Batch, C]) -> float32 { grad { ... } }
        let mut stmt = Stmt::new(StmtKind::TrainDecl, tok);

        self.expect(TokenType::OpLparen, "expected ( after train")?;
        while !self.check(TokenType::OpRparen) {
            let mut param = Param {
                name: String::new(),
                type_: None,
                is_mut: false,
                is_ref: false,
            };
            param.is_mut = self.match_(TokenType::KwMut);
            param.is_ref = self.match_(TokenType::KwRef);
            let pname = self.expect(TokenType::Identifier, "Expected parameter name")?;
            param.name = pname.value;
            if self.match_(TokenType::OpColon) {
                param.type_ = Some(Box::new(self.parse_type()?));
            }
            stmt.params.push(param);
            if !self.match_(TokenType::OpComma) {
                break;
            }
        }
        self.expect(TokenType::OpRparen, "expected ) after train params")?;

        if self.match_(TokenType::OpPipeline) || self.match_(TokenType::OpColon) {
            stmt.return_type = Some(Box::new(self.parse_type()?));
        }

        stmt.body = Some(Box::new(self.parse_block()?));
        Ok(stmt)
    }

    fn parse_network_decl(&mut self, tok: Token) -> NsResult<Stmt> {
        let mut stmt = Stmt::new(StmtKind::NetworkDecl, tok);

        let name = self.expect(TokenType::Identifier, "Expected network name")?;
        stmt.network_name = name.value;

        self.expect(TokenType::OpLbrace, "expected { after network name")?;

        while !self.check(TokenType::OpRbrace) {
            // Detect input/output sections: <name> ':' Tensor[...]  (or <name> ':')
            if self.check(TokenType::Identifier)
                && self.peek_ahead(1).type_ == TokenType::OpColon
                && self.peek_ahead(2).type_ == TokenType::TypeTensor
            {
                let io = self.advance();
                self.advance(); // consume ':'
                let mut sub = Stmt::new(StmtKind::VarDecl, io.clone());
                sub.var_name = io.value;
                sub.var_type = Some(Box::new(self.parse_type()?));
                self.match_(TokenType::OpSemicolon);
                stmt.methods.push(Box::new(sub));
            } else if self.check(TokenType::KwLayer) {
                self.advance();
                stmt.layers.push(Box::new(self.parse_layer_decl(self.previous().clone())?));
            } else if self.check(TokenType::KwForward) {
                stmt.methods.push(Box::new(self.parse_statement()?));
            } else if self.check(TokenType::KwTrain) {
                stmt.methods.push(Box::new(self.parse_statement()?));
            } else if self.check(TokenType::KwType) {
                self.advance();
                stmt.methods.push(Box::new(self.parse_type_decl(self.previous().clone())?));
            } else {
                // treat as general statement
                stmt.methods.push(Box::new(self.parse_statement()?));
            }
        }
        self.expect(TokenType::OpRbrace, "expected } after network")?;

        Ok(stmt)
    }

    fn parse_layer_decl(&mut self, tok: Token) -> NsResult<Stmt> {
        // layer fc1 = Dense(in: InDim, out: 512, activation: ReLU)
        let mut stmt = Stmt::new(StmtKind::LayerDecl, tok);

        let name = self.expect(TokenType::Identifier, "Expected layer name")?;
        stmt.layer_name = name.value;

        self.expect(TokenType::OpAssign, "expected '=' after layer name")?;

        let layer_type = self.expect(TokenType::Identifier, "Expected layer type (Dense, Dropout, etc.)")?;
        stmt.layer_type = layer_type.value;

        if self.match_(TokenType::OpLparen) {
            while !self.check(TokenType::OpRparen) {
                let lp_name = self.expect(TokenType::Identifier, "Expected parameter name")?;
                let name = lp_name.value;
                self.expect(TokenType::OpColon, "expected ':' after layer param name")?;
                let value = self.parse_expression()?;
                stmt.layer_params.push(super::ast::LayerParam {
                    name,
                    value: Some(Box::new(value)),
                });
                if !self.match_(TokenType::OpComma) {
                    break;
                }
            }
            self.expect(TokenType::OpRparen, "expected ) after layer params")?;
        }

        self.match_(TokenType::OpSemicolon);
        Ok(stmt)
    }

    fn parse_grad_block(&mut self, tok: Token) -> NsResult<Stmt> {
        let mut stmt = Stmt::new(StmtKind::GradBlock, tok);
        if self.match_(TokenType::OpLparen) {
            while !self.check(TokenType::OpRparen) {
                let p = self.advance();
                stmt.params.push(Param {
                    name: p.value,
                    type_: None,
                    is_mut: false,
                    is_ref: false,
                });
                if !self.match_(TokenType::OpComma) {
                    break;
                }
            }
            self.expect(TokenType::OpRparen, "expected ) after grad params")?;
        }
        stmt.grad_body = Some(Box::new(self.parse_block()?));
        Ok(stmt)
    }

    fn parse_block(&mut self) -> NsResult<Stmt> {
        let tok = self.expect(TokenType::OpLbrace, "expected { to start block")?;
        let mut stmt = Stmt::new(StmtKind::Block, tok);
        while !self.check(TokenType::OpRbrace) && !self.at_end() {
            stmt.statements.push(Box::new(self.parse_statement()?));
        }
        self.expect(TokenType::OpRbrace, "expected } to close block")?;
        Ok(stmt)
    }

    fn parse_return_stmt(&mut self) -> NsResult<Stmt> {
        let tok = self.advance();
        let mut stmt = Stmt::new(StmtKind::ReturnStmt, tok);
        if !self.check(TokenType::OpSemicolon) && !self.check(TokenType::OpRbrace) {
            stmt.init_expr = Some(Box::new(self.parse_expression()?));
        }
        self.match_(TokenType::OpSemicolon);
        Ok(stmt)
    }

    fn parse_if_stmt(&mut self) -> NsResult<Stmt> {
        let tok = self.advance();
        let mut stmt = Stmt::new(StmtKind::IfStmt, tok);
        stmt.condition = Some(Box::new(self.parse_expression()?));
        stmt.body = Some(Box::new(self.parse_statement()?));
        if self.match_(TokenType::KwElse) {
            stmt.else_branch = Some(Box::new(self.parse_statement()?));
        }
        Ok(stmt)
    }

    fn parse_while_stmt(&mut self) -> NsResult<Stmt> {
        let tok = self.advance();
        let mut stmt = Stmt::new(StmtKind::WhileStmt, tok);
        stmt.condition = Some(Box::new(self.parse_expression()?));
        stmt.body = Some(Box::new(self.parse_statement()?));
        Ok(stmt)
    }

    fn parse_expr_stmt(&mut self) -> NsResult<Stmt> {
        let tok = self.peek().clone();
        let mut stmt = Stmt::new(StmtKind::ExprStmt, tok);
        stmt.expr = Some(Box::new(self.parse_expression()?));
        self.match_(TokenType::OpSemicolon);
        Ok(stmt)
    }

    fn parse_type_decl(&mut self, tok: Token) -> NsResult<Stmt> {
        // type Batch = Dynamic
        // type Features = 784
        let mut stmt = Stmt::new(StmtKind::TypeDecl, tok);
        let name = self.expect(TokenType::Identifier, "Expected type alias name")?;
        stmt.alias_name = name.value;
        self.expect(TokenType::OpAssign, "expected '=' in type declaration")?;
        stmt.alias_expr = Some(Box::new(self.parse_primary()?));
        self.match_(TokenType::OpSemicolon);
        Ok(stmt)
    }

    // ---- Expressions ----

    fn parse_expression(&mut self) -> NsResult<Expr> {
        self.parse_assignment()
    }

    fn parse_assignment(&mut self) -> NsResult<Expr> {
        let expr = self.parse_pipeline()?;
        if self.match_(TokenType::OpAssign) {
            let mut assign = Expr::new(ExprKind::BinaryOp, self.previous().clone());
            assign.left = Some(Box::new(expr));
            assign.right = Some(Box::new(self.parse_assignment()?));
            return Ok(assign);
        }
        Ok(expr)
    }

    fn parse_pipeline(&mut self) -> NsResult<Expr> {
        let mut left = self.parse_or()?;
        while self.match_(TokenType::OpPipeline) {
            let mut pip = Expr::new(ExprKind::PipelineOp, self.previous().clone());
            pip.left = Some(Box::new(left));
            pip.right = Some(Box::new(self.parse_or()?));
            left = pip;
        }
        Ok(left)
    }

    fn parse_or(&mut self) -> NsResult<Expr> {
        let mut left = self.parse_and()?;
        while self.match_(TokenType::OpOr) {
            let mut expr = Expr::new(ExprKind::BinaryOp, self.previous().clone());
            expr.left = Some(Box::new(left));
            expr.right = Some(Box::new(self.parse_and()?));
            left = expr;
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> NsResult<Expr> {
        let mut left = self.parse_equality()?;
        while self.match_(TokenType::OpAnd) {
            let mut expr = Expr::new(ExprKind::BinaryOp, self.previous().clone());
            expr.left = Some(Box::new(left));
            expr.right = Some(Box::new(self.parse_equality()?));
            left = expr;
        }
        Ok(left)
    }

    fn parse_equality(&mut self) -> NsResult<Expr> {
        let mut left = self.parse_comparison()?;
        while self.match_one_of(&[TokenType::OpEq, TokenType::OpNeq]) {
            let mut expr = Expr::new(ExprKind::BinaryOp, self.previous().clone());
            expr.left = Some(Box::new(left));
            expr.right = Some(Box::new(self.parse_comparison()?));
            left = expr;
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> NsResult<Expr> {
        let mut left = self.parse_additive()?;
        while self.match_one_of(&[TokenType::OpLt, TokenType::OpGt, TokenType::OpLte, TokenType::OpGte]) {
            let mut expr = Expr::new(ExprKind::BinaryOp, self.previous().clone());
            expr.left = Some(Box::new(left));
            expr.right = Some(Box::new(self.parse_additive()?));
            left = expr;
        }
        Ok(left)
    }

    fn parse_additive(&mut self) -> NsResult<Expr> {
        let mut left = self.parse_multiplicative()?;
        while self.match_one_of(&[TokenType::OpPlus, TokenType::OpMinus]) {
            let mut expr = Expr::new(ExprKind::BinaryOp, self.previous().clone());
            expr.left = Some(Box::new(left));
            expr.right = Some(Box::new(self.parse_multiplicative()?));
            left = expr;
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> NsResult<Expr> {
        let mut left = self.parse_matmul()?;
        while self.match_one_of(&[TokenType::OpStar, TokenType::OpSlash, TokenType::OpPercent]) {
            let mut expr = Expr::new(ExprKind::BinaryOp, self.previous().clone());
            expr.left = Some(Box::new(left));
            expr.right = Some(Box::new(self.parse_matmul()?));
            left = expr;
        }
        Ok(left)
    }

    fn parse_matmul(&mut self) -> NsResult<Expr> {
        let mut left = self.parse_unary()?;
        while self.match_(TokenType::OpMatmul) {
            let mut expr = Expr::new(ExprKind::MatmulOp, self.previous().clone());
            expr.left = Some(Box::new(left));
            expr.right = Some(Box::new(self.parse_unary()?));
            left = expr;
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> NsResult<Expr> {
        if self.match_one_of(&[TokenType::OpNot, TokenType::OpMinus]) {
            let mut expr = Expr::new(ExprKind::UnaryOp, self.previous().clone());
            expr.operand = Some(Box::new(self.parse_unary()?));
            return Ok(expr);
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> NsResult<Expr> {
        let mut expr = self.parse_primary()?;

        loop {
            if self.match_(TokenType::OpLparen) {
                expr = self.parse_function_call(expr)?;
            } else if self.match_(TokenType::OpLbracket) {
                let mut index_expr = Expr::new(ExprKind::IndexOp, self.previous().clone());
                index_expr.operand = Some(Box::new(expr));
                if !self.check(TokenType::OpRbracket) {
                    loop {
                        index_expr.indices.push(Box::new(self.parse_expression()?));
                        if !self.match_(TokenType::OpComma) {
                            break;
                        }
                        if self.check(TokenType::OpRbracket) {
                            break;
                        }
                    }
                }
                self.expect(TokenType::OpRbracket, "expected ] after index")?;
                expr = index_expr;
            } else if self.match_(TokenType::OpDot) {
                // Member access or method call: model.forward(...) / opt.step(model)
                if !self.check(TokenType::Identifier)
                    && !self.check(TokenType::KwForward)
                    && !self.check(TokenType::KwTrain)
                {
                    return Err(self.error(self.peek(), "expected member name after '.'"));
                }
                let member = self.advance();
                let member_expr = Expr::new(ExprKind::Identifier, member);
                if self.match_(TokenType::OpLparen) {
                    let mut call = Expr::new(ExprKind::FunctionCall, member_expr.token.clone());
                    call.operand = Some(Box::new(member_expr));
                    if !self.check(TokenType::OpRparen) {
                        loop {
                            call.args.push(Box::new(self.parse_expression()?));
                            if !self.match_(TokenType::OpComma) {
                                break;
                            }
                            if self.check(TokenType::OpRparen) {
                                break;
                            }
                        }
                    }
                    self.expect(TokenType::OpRparen, "expected ) after method arguments")?;
                    expr = call;
                } else {
                    expr = member_expr;
                }
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> NsResult<Expr> {
        let tok = self.peek().clone();

        match tok.type_ {
            TokenType::IntLiteral => {
                self.advance();
                Ok(Expr::new(ExprKind::LiteralInt, tok))
            }
            TokenType::FloatLiteral => {
                self.advance();
                Ok(Expr::new(ExprKind::LiteralFloat, tok))
            }
            TokenType::StringLiteral => {
                self.advance();
                Ok(Expr::new(ExprKind::LiteralString, tok))
            }
            TokenType::Identifier => {
                self.advance();
                // bool literals
                if tok.value == "true" || tok.value == "false" {
                    Ok(Expr::new(ExprKind::LiteralBool, tok))
                } else {
                    Ok(Expr::new(ExprKind::Identifier, tok))
                }
            }
            TokenType::OpLparen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(TokenType::OpRparen, "expected ) after expression")?;
                Ok(expr)
            }
            TokenType::OpMinus => self.parse_unary(),
            TokenType::TypeDynamic => {
                self.advance();
                Ok(Expr::new(ExprKind::Identifier, tok))
            }
            TokenType::ActRelu
            | TokenType::ActLeakyRelu
            | TokenType::ActSigmoid
            | TokenType::ActTanh
            | TokenType::ActSwish
            | TokenType::ActGelu
            | TokenType::ActSwiglu
            | TokenType::ActSilu
            | TokenType::ActIdentity
            | TokenType::ActSoftmax => {
                self.advance();
                Ok(Expr::new(ExprKind::Identifier, tok))
            }
            TokenType::OptAdamw | TokenType::OptMuon | TokenType::OptSgd => {
                self.advance();
                Ok(Expr::new(ExprKind::Identifier, tok))
            }
            _ => Err(self.error(&tok, "Unexpected token in expression")),
        }
    }

    fn parse_function_call(&mut self, callee: Expr) -> NsResult<Expr> {
        let mut expr = Expr::new(ExprKind::FunctionCall, self.previous().clone());
        expr.operand = Some(Box::new(callee));

        if !self.check(TokenType::OpRparen) {
            loop {
                expr.args.push(Box::new(self.parse_expression()?));
                if !self.match_(TokenType::OpComma) {
                    break;
                }
                if self.check(TokenType::OpRparen) {
                    break;
                }
            }
        }
        self.expect(TokenType::OpRparen, "expected ) after function arguments")?;
        Ok(expr)
    }

    // ---- Types ----

    fn parse_type(&mut self) -> NsResult<TypeNode> {
        if self.check(TokenType::TypeTensor) {
            let tt = self.parse_tensor_type()?;
            return Ok(TypeNode::tensor(tt));
        }
        if self.is_dtype(self.peek().type_) {
            let tt = self.advance().type_;
            return Ok(TypeNode::scalar(Dtype::from_token(tt)));
        }
        if self.check(TokenType::TypeInt)
            || self.check(TokenType::TypeFloat)
            || self.check(TokenType::TypeBool)
            || self.check(TokenType::TypeString)
        {
            let dt = match self.advance().type_ {
                TokenType::TypeInt => Dtype::Int64,
                TokenType::TypeFloat => Dtype::Float64,
                TokenType::TypeBool => Dtype::Bool,
                _ => Dtype::Int64,
            };
            return Ok(TypeNode::scalar(dt));
        }
        let tok = self.peek().clone();
        Err(self.error(&tok, "Expected type"))
    }

    fn parse_tensor_type(&mut self) -> NsResult<TensorType> {
        self.expect(TokenType::TypeTensor, "expected Tensor")?;
        self.expect(TokenType::OpLbracket, "expected [ after Tensor")?;

        let mut dims: Vec<DimExpr> = Vec::new();
        loop {
            if self.check(TokenType::TypeDynamic) || self.check(TokenType::Identifier) {
                let t = self.advance();
                if t.value == "Dynamic" {
                    dims.push(DimExpr::dynamic());
                } else {
                    dims.push(DimExpr::symbolic(&t.value));
                }
            } else if self.check(TokenType::IntLiteral) {
                let t = self.advance();
                let v: i64 = t.value.parse().map_err(|_| {
                    NsError(format!("Invalid dimension literal '{}'", t.value))
                })?;
                dims.push(DimExpr::constant(v));
            } else {
                return Err(self.error(self.peek(), "Expected dimension (constant, identifier, or Dynamic)"));
            }
            if !self.match_(TokenType::OpComma) {
                break;
            }
            if self.check(TokenType::OpRbracket) {
                break;
            }
        }

        self.expect(TokenType::OpRbracket, "expected ] after tensor dims")?;

        let mut dtype = Dtype::Float32;
        if self.is_dtype(self.peek().type_) {
            dtype = Dtype::from_token(self.advance().type_);
        }

        Ok(TensorType::new(dims, dtype))
    }

    fn parse_dtype(&mut self) -> NsResult<Dtype> {
        if !self.is_dtype(self.peek().type_) {
            return Err(self.error(self.peek(), "Expected dtype"));
        }
        Ok(Dtype::from_token(self.advance().type_))
    }

    fn is_dtype(&self, tt: TokenType) -> bool {
        matches!(
            tt,
            TokenType::DtypeFloat16
                | TokenType::DtypeFloat32
                | TokenType::DtypeFloat64
                | TokenType::DtypeInt8
                | TokenType::DtypeInt16
                | TokenType::DtypeInt32
                | TokenType::DtypeInt64
                | TokenType::DtypeFp8
                | TokenType::DtypeFp4
                | TokenType::DtypeBool
        )
    }
}

/// Convenience: parse a `.ns` source string straight to a `Program`.
pub fn parse_source(source: &str) -> NsResult<Program> {
    let mut lexer = super::lexer::Lexer::new(source);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}