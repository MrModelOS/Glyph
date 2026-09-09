use crate::ast::*;
use crate::lexer::{SpannedToken, Token, LexerError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("Unexpected token {0:?} at line {1}:{2}, expected {3}")]
    UnexpectedToken(Token, usize, usize, String),

    #[error("Lexer error: {0}")]
    LexerError(#[from] LexerError),
}

pub struct Parser {
    tokens: Vec<SpannedToken>,
    pos: usize,
    active_type_params: Vec<String>,
}

struct FunctionRest {
    params: Vec<FunctionParam>,
    return_type: Option<Type>,
    body: Vec<Stmt>,
    is_async: bool,
}

impl Parser {
    pub fn new(tokens: Vec<SpannedToken>) -> Self {
        Parser {
            tokens,
            pos: 0,
            active_type_params: Vec::new(),
        }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).map(|t| &t.token).unwrap_or(&Token::Eof)
    }

    fn peek_spanned(&self) -> &SpannedToken {
        self.tokens.get(self.pos).unwrap_or(&SpannedToken {
            token: Token::Eof,
            line: 0,
            column: 0,
            end_line: 0,
            end_column: 0,
        })
    }

    /// End position (exclusive) of the most recently consumed token.
    fn last_end(&self) -> LineCol {
        if self.pos == 0 {
            return LineCol { line: 1, col: 1 };
        }
        let t = &self.tokens[self.pos - 1];
        LineCol { line: t.end_line, col: t.end_column }
    }

    /// Build a span from a token that begins the node to the current end.
    fn span_of(&self, start: &SpannedToken) -> Span {
        Span::new(
            LineCol { line: start.line, col: start.column },
            self.last_end(),
        )
    }

    fn advance(&mut self) -> &SpannedToken {
        let token = &self.tokens[self.pos];
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        token
    }

    fn expect(&mut self, expected: &Token) -> Result<(), ParseError> {
        let spanned = self.peek_spanned();
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(expected) {
            self.advance();
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken(
                spanned.token.clone(),
                spanned.line,
                spanned.column,
                format!("{:?}", expected),
            ))
        }
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        let spanned = self.peek_spanned().clone();
        match &spanned.token {
            Token::Identifier(name) => {
                self.advance();
                Ok(name.clone())
            }
            _ => Err(ParseError::UnexpectedToken(
                spanned.token.clone(),
                spanned.line,
                spanned.column,
                "identifier".to_string(),
            )),
        }
    }

    /// Expect an identifier and return it together with the exact span of the
    /// name token (used for definition lookup / go-to-definition).
    fn expect_ident_span(&mut self) -> Result<(String, Span), ParseError> {
        let spanned = self.peek_spanned().clone();
        match &spanned.token {
            Token::Identifier(name) => {
                self.advance();
                let span = Span::new(
                    LineCol { line: spanned.line, col: spanned.column },
                    LineCol {
                        line: spanned.end_line,
                        col: spanned.end_column,
                    },
                );
                Ok((name.clone(), span))
            }
            _ => Err(ParseError::UnexpectedToken(
                spanned.token.clone(),
                spanned.line,
                spanned.column,
                "identifier".to_string(),
            )),
        }
    }

    pub fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut items = Vec::new();

        while self.peek() != &Token::Eof {
            items.push(self.parse_top_level_item()?);
        }

        Ok(Program { items })
    }

    fn parse_top_level_item(&mut self) -> Result<TopLevelItem, ParseError> {
        // Check for @pub visibility and @test attributes (any order)
        let mut pub_vis = Visibility::Private;
        let mut is_test = false;
        loop {
            match self.peek() {
                Token::Pub => {
                    self.advance(); // consume @pub
                    pub_vis = Visibility::Public;
                }
                Token::Test => {
                    self.advance(); // consume @test
                    is_test = true;
                }
                _ => break,
            }
        }

        match self.peek() {
            Token::Module => self.parse_module(),
            Token::Use => self.parse_use(),
            Token::Fn => self.parse_function_with_vis(pub_vis, is_test),
            Token::Struct => self.parse_struct_with_vis(pub_vis),
            Token::Enum => self.parse_enum_with_vis(pub_vis),
            Token::Trait => self.parse_trait_with_vis(pub_vis),
            Token::Impl => self.parse_impl_with_vis(pub_vis),
            Token::Const => self.parse_const_with_vis(pub_vis),
            _ => {
                let spanned = self.peek_spanned().clone();
                Err(ParseError::UnexpectedToken(
                    spanned.token.clone(),
                    spanned.line,
                    spanned.column,
                    "top-level item (@fn, @struct, @enum, @trait, @impl, @const, @use, @pub)".to_string(),
                ))
            }
        }
    }

    fn parse_module(&mut self) -> Result<TopLevelItem, ParseError> {
        self.advance(); // consume @module
        let mut path = Vec::new();
        path.push(self.expect_ident()?);

        while self.peek() == &Token::DoubleColon || self.peek() == &Token::Dot {
            self.advance();
            path.push(self.expect_ident()?);
        }

        Ok(TopLevelItem::Module { path })
    }

    fn parse_use(&mut self) -> Result<TopLevelItem, ParseError> {
        self.advance(); // consume @use
        let mut path = Vec::new();
        path.push(self.expect_ident()?);

        while self.peek() == &Token::DoubleColon || self.peek() == &Token::Dot {
            self.advance();
            path.push(self.expect_ident()?);
        }

        self.expect(&Token::Semicolon)?;

        Ok(TopLevelItem::Use { path })
    }

    fn parse_function_with_vis(
        &mut self,
        pub_vis: Visibility,
        is_test: bool,
    ) -> Result<TopLevelItem, ParseError> {
        self.advance(); // consume @fn

        let is_async = if self.peek() == &Token::Async {
            self.advance();
            true
        } else {
            false
        };

        let (name, name_span) = self.expect_ident_span()?;

        // Optional type parameters: <T, K>
        let mut type_params = Vec::new();
        if self.peek() == &Token::Lt {
            self.advance();
            loop {
                type_params.push(self.expect_ident()?);
                if self.peek() == &Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
            self.expect(&Token::Gt)?;
        }

        let outer_params = std::mem::replace(&mut self.active_type_params, type_params.clone());
        let result = self.parse_function_rest(&name, is_async);
        self.active_type_params = outer_params;
        let rest = result?;

        Ok(TopLevelItem::Function {
            name,
            name_span,
            type_params,
            params: rest.params,
            return_type: rest.return_type,
            body: rest.body,
            is_async: rest.is_async,
            is_test,
            pub_vis,
        })
    }

    fn parse_function_rest(
        &mut self,
        _name: &str,
        is_async: bool,
    ) -> Result<FunctionRest, ParseError> {
        self.expect(&Token::LParen)?;

        let mut params = Vec::new();
        if self.peek() != &Token::RParen {
            loop {
                let is_move = if let Token::Identifier(ref s) = self.peek() {
                    if s == "move" {
                        self.advance();
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

                let (param_name, param_span) = self.expect_ident_span()?;
                self.expect(&Token::Colon)?;
                let param_type = self.parse_type()?;

                params.push(FunctionParam {
                    name: param_name,
                    name_span: param_span,
                    ty: param_type,
                    is_move,
                });

                if self.peek() == &Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        self.expect(&Token::RParen)?;

        let return_type = if self.peek() == &Token::Arrow {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(&Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(&Token::RBrace)?;

        Ok(FunctionRest {
            params,
            return_type,
            body,
            is_async,
        })
    }

    fn parse_struct_with_vis(&mut self, pub_vis: Visibility) -> Result<TopLevelItem, ParseError> {
        self.advance(); // consume @struct
        let (name, name_span) = self.expect_ident_span()?;
        self.expect(&Token::LBrace)?;

        let mut fields = Vec::new();
        while self.peek() != &Token::RBrace {
            let field_name = self.expect_ident()?;
            self.expect(&Token::Colon)?;
            let field_type = self.parse_type()?;
            fields.push((field_name, field_type));

            if self.peek() == &Token::Comma {
                self.advance();
            }
        }

        self.expect(&Token::RBrace)?;

        Ok(TopLevelItem::Struct { name, name_span, fields, pub_vis })
    }

    fn parse_enum_with_vis(&mut self, pub_vis: Visibility) -> Result<TopLevelItem, ParseError> {
        self.advance(); // consume @enum
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;

        let mut variants = Vec::new();
        while self.peek() != &Token::RBrace {
            let variant_name = self.expect_ident()?;

            // Parse optional data: Variant(Type1, Type2)
            let data = if self.peek() == &Token::LParen {
                self.advance();
                let mut types = Vec::new();
                if self.peek() != &Token::RParen {
                    loop {
                        types.push(self.parse_type()?);
                        if self.peek() == &Token::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&Token::RParen)?;
                Some(types)
            } else {
                None
            };

            let value = if self.peek() == &Token::Assign {
                self.advance();
                match self.peek().clone() {
                    Token::Integer(v) => {
                        self.advance();
                        Some(v)
                    }
                    _ => {
                        let spanned = self.peek_spanned().clone();
                        return Err(ParseError::UnexpectedToken(
                            spanned.token.clone(),
                            spanned.line,
                            spanned.column,
                            "integer literal".to_string(),
                        ));
                    }
                }
            } else {
                None
            };

            variants.push(EnumVariant {
                name: variant_name,
                value,
                data,
            });

            if self.peek() == &Token::Comma {
                self.advance();
            }
        }

        self.expect(&Token::RBrace)?;

        Ok(TopLevelItem::Enum { name, variants, pub_vis })
    }

    fn parse_trait_with_vis(&mut self, pub_vis: Visibility) -> Result<TopLevelItem, ParseError> {
        self.advance(); // consume @trait
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;

        let mut methods = Vec::new();
        while self.peek() != &Token::RBrace {
            self.expect(&Token::Fn)?;
            let method_name = self.expect_ident()?;
            self.expect(&Token::LParen)?;

            let mut params = Vec::new();
            if self.peek() != &Token::RParen {
                loop {
                    let (param_name, param_span) = self.expect_ident_span()?;
                    self.expect(&Token::Colon)?;
                    let param_type = self.parse_type()?;
                    params.push(FunctionParam {
                        name: param_name,
                        name_span: param_span,
                        ty: param_type,
                        is_move: false,
                    });

                    if self.peek() == &Token::Comma {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }

            self.expect(&Token::RParen)?;

            let return_type = if self.peek() == &Token::Arrow {
                self.advance();
                Some(self.parse_type()?)
            } else {
                None
            };

            self.expect(&Token::Semicolon)?;

            methods.push(TraitMethod {
                name: method_name,
                params,
                return_type,
            });
        }

        self.expect(&Token::RBrace)?;

        Ok(TopLevelItem::Trait { name, methods, pub_vis })
    }

    fn parse_impl_with_vis(&mut self, pub_vis: Visibility) -> Result<TopLevelItem, ParseError> {
        self.advance(); // consume @impl
        let type_name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;

        let mut methods = Vec::new();
        while self.peek() != &Token::RBrace {
            self.expect(&Token::Fn)?;
            let is_async = if self.peek() == &Token::Async {
                self.advance();
                true
            } else {
                false
            };
            let method_name = self.expect_ident()?;
            self.expect(&Token::LParen)?;

            let mut params = Vec::new();
            if self.peek() != &Token::RParen {
                loop {
                    let (param_name, param_span) = self.expect_ident_span()?;
                    self.expect(&Token::Colon)?;
                    let param_type = self.parse_type()?;
                    params.push(FunctionParam {
                        name: param_name,
                        name_span: param_span,
                        ty: param_type,
                        is_move: false,
                    });

                    if self.peek() == &Token::Comma {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }

            self.expect(&Token::RParen)?;

            let return_type = if self.peek() == &Token::Arrow {
                self.advance();
                Some(self.parse_type()?)
            } else {
                None
            };

            self.expect(&Token::LBrace)?;
            let body = self.parse_block()?;
            self.expect(&Token::RBrace)?;

            methods.push(ImplMethod {
                name: method_name,
                params,
                return_type,
                body,
                is_async,
            });
        }

        self.expect(&Token::RBrace)?;

        Ok(TopLevelItem::Impl { type_name, methods, pub_vis })
    }

    fn parse_const_with_vis(&mut self, pub_vis: Visibility) -> Result<TopLevelItem, ParseError> {
        self.advance(); // consume @const
        let name = self.expect_ident()?;
        self.expect(&Token::Colon)?;
        let ty = self.parse_type()?;
        self.expect(&Token::Assign)?;
        let value = self.parse_expression()?;
        self.expect(&Token::Semicolon)?;

        Ok(TopLevelItem::Const { name, ty, value, pub_vis })
    }

    fn parse_type(&mut self) -> Result<Type, ParseError> {
        let spanned = self.peek_spanned().clone();
        match &spanned.token {
            Token::Ref => {
                self.advance();
                let inner = self.parse_type()?;
                Ok(Type::Ref(Box::new(inner)))
            }
            Token::TypeString => {
                self.advance();
                Ok(Type::String)
            }
            Token::TypeUInt64 => {
                self.advance();
                Ok(Type::UInt64)
            }
            Token::TypeInt64 => {
                self.advance();
                Ok(Type::Int64)
            }
            Token::TypeFloat64 => {
                self.advance();
                Ok(Type::Float64)
            }
            Token::TypeBool => {
                self.advance();
                Ok(Type::Bool)
            }
            Token::TypeVoid => {
                self.advance();
                Ok(Type::Void)
            }
            Token::TypeBytes => {
                self.advance();
                Ok(Type::Bytes)
            }
            Token::TypeList => {
                self.advance();
                self.expect(&Token::Lt)?;
                let inner = self.parse_type()?;
                self.expect(&Token::Gt)?;
                Ok(Type::List(Box::new(inner)))
            }
            Token::TypeMap => {
                self.advance();
                self.expect(&Token::Lt)?;
                let key = self.parse_type()?;
                self.expect(&Token::Comma)?;
                let value = self.parse_type()?;
                self.expect(&Token::Gt)?;
                Ok(Type::Map(Box::new(key), Box::new(value)))
            }
            Token::TypeResult => {
                self.advance();
                self.expect(&Token::Lt)?;
                let ok_type = self.parse_type()?;
                self.expect(&Token::Comma)?;
                let err_type = self.parse_type()?;
                self.expect(&Token::Gt)?;
                Ok(Type::Result(Box::new(ok_type), Box::new(err_type)))
            }
            Token::TypeOption => {
                self.advance();
                self.expect(&Token::Lt)?;
                let inner = self.parse_type()?;
                self.expect(&Token::Gt)?;
                Ok(Type::Option(Box::new(inner)))
            }
            Token::TypeAsync => {
                self.advance();
                self.expect(&Token::Lt)?;
                let inner = self.parse_type()?;
                self.expect(&Token::Gt)?;
                Ok(Type::Async(Box::new(inner)))
            }
            Token::TypeChannel => {
                self.advance();
                self.expect(&Token::Lt)?;
                let inner = self.parse_type()?;
                self.expect(&Token::Gt)?;
                Ok(Type::Channel(Box::new(inner)))
            }
            Token::LBracket => {
                // Array type [Type; size]
                self.advance();
                let inner = self.parse_type()?;
                self.expect(&Token::Semicolon)?;
                let size = match self.peek().clone() {
                    Token::Integer(v) => {
                        self.advance();
                        v as usize
                    }
                    _ => {
                        let spanned = self.peek_spanned().clone();
                        return Err(ParseError::UnexpectedToken(
                            spanned.token.clone(),
                            spanned.line,
                            spanned.column,
                            "integer size".to_string(),
                        ));
                    }
                };
                self.expect(&Token::RBracket)?;
                Ok(Type::Array(Box::new(inner), size))
            }
            Token::Identifier(name) => {
                if self.active_type_params.iter().any(|p| p == name) {
                    self.advance();
                    Ok(Type::Generic(name.clone()))
                } else {
                    self.advance();
                    Ok(Type::Custom(name.clone()))
                }
            }
            _ => Err(ParseError::UnexpectedToken(
                spanned.token.clone(),
                spanned.line,
                spanned.column,
                "type".to_string(),
            )),
        }
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, ParseError> {
        let mut stmts = Vec::new();

        while self.peek() != &Token::RBrace && self.peek() != &Token::Eof {
            stmts.push(self.parse_statement()?);
        }

        Ok(stmts)
    }

    fn parse_statement(&mut self) -> Result<Stmt, ParseError> {
        match self.peek() {
            Token::Let => self.parse_let(),
            Token::Return => self.parse_return(),
            Token::Break => {
                let loc = LineCol { line: self.peek_spanned().line, col: self.peek_spanned().column };
                self.advance();
                self.expect(&Token::Semicolon)?;
                Ok(Stmt::Break(loc))
            }
            Token::Continue => {
                let loc = LineCol { line: self.peek_spanned().line, col: self.peek_spanned().column };
                self.advance();
                self.expect(&Token::Semicolon)?;
                Ok(Stmt::Continue(loc))
            }
            Token::Loop => self.parse_loop(),
            Token::While => self.parse_while(),
            Token::For => self.parse_for(),
            Token::Guard => self.parse_guard(),
            Token::Spawn => self.parse_spawn(),
            Token::Select => self.parse_select(),
            Token::If => {
                let loc = LineCol { line: self.peek_spanned().line, col: self.peek_spanned().column };
                let expr = self.parse_if()?;
                Ok(Stmt::Expression(loc, expr))
            }
            Token::Match => {
                let loc = LineCol { line: self.peek_spanned().line, col: self.peek_spanned().column };
                let expr = self.parse_match()?;
                Ok(Stmt::Expression(loc, expr))
            }
            _ => self.parse_expression_statement(),
        }
    }

    fn parse_let(&mut self) -> Result<Stmt, ParseError> {
        let loc = LineCol { line: self.peek_spanned().line, col: self.peek_spanned().column };
        self.advance(); // consume let

        let mutable = if self.peek() == &Token::Mut {
            self.advance();
            true
        } else {
            false
        };

        let (name, name_span) = self.expect_ident_span()?;

        let ty = if self.peek() == &Token::Colon {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(&Token::Assign)?;
        let value = self.parse_expression()?;
        self.expect(&Token::Semicolon)?;

        Ok(Stmt::Let {
            loc,
            name,
            name_span,
            ty,
            value,
            mutable,
        })
    }

    fn parse_return(&mut self) -> Result<Stmt, ParseError> {
        let loc = LineCol { line: self.peek_spanned().line, col: self.peek_spanned().column };
        self.advance(); // consume return

        let value = if self.peek() == &Token::Semicolon {
            None
        } else {
            Some(self.parse_expression()?)
        };

        self.expect(&Token::Semicolon)?;

        Ok(Stmt::Return(loc, value))
    }

    fn parse_loop(&mut self) -> Result<Stmt, ParseError> {
        let loc = LineCol { line: self.peek_spanned().line, col: self.peek_spanned().column };
        self.advance(); // consume loop
        self.expect(&Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(&Token::RBrace)?;

        Ok(Stmt::Loop(loc, body))
    }

    fn parse_while(&mut self) -> Result<Stmt, ParseError> {
        let loc = LineCol { line: self.peek_spanned().line, col: self.peek_spanned().column };
        self.advance(); // consume while
        let condition = self.parse_expression()?;
        self.expect(&Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(&Token::RBrace)?;

        Ok(Stmt::While {
            loc,
            condition,
            body,
        })
    }

    fn parse_for(&mut self) -> Result<Stmt, ParseError> {
        let loc = LineCol { line: self.peek_spanned().line, col: self.peek_spanned().column };
        self.advance(); // consume for
        let spanned_for_var = self.peek_spanned().clone();
        let variable = self.expect_ident()?;
        self.expect(&Token::In)?;
        let iterable = self.parse_expression()?;
        self.expect(&Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(&Token::RBrace)?;

        Ok(Stmt::For {
            loc,
            variable: variable.clone(),
            var_span: Span::new(
                LineCol { line: spanned_for_var.line, col: spanned_for_var.column },
                LineCol {
                    line: spanned_for_var.end_line,
                    col: spanned_for_var.end_column,
                },
            ),
            iterable,
            body,
        })
    }

    fn parse_guard(&mut self) -> Result<Stmt, ParseError> {
        let loc = LineCol { line: self.peek_spanned().line, col: self.peek_spanned().column };
        self.advance(); // consume #guard
        self.expect(&Token::LParen)?;
        let condition = self.parse_expression()?;
        self.expect(&Token::RParen)?;

        let mut else_body = Vec::new();
        if self.peek() == &Token::Else {
            self.advance();
            self.expect(&Token::LBrace)?;
            else_body = self.parse_block()?;
            self.expect(&Token::RBrace)?;
        }

        self.expect(&Token::Semicolon)?;

        Ok(Stmt::Guard {
            loc,
            condition,
            else_body,
        })
    }

    fn parse_spawn(&mut self) -> Result<Stmt, ParseError> {
        let loc = LineCol { line: self.peek_spanned().line, col: self.peek_spanned().column };
        self.advance(); // consume spawn
        let expr = self.parse_expression()?;
        self.expect(&Token::Semicolon)?;

        Ok(Stmt::Spawn(loc, expr))
    }

    fn parse_select(&mut self) -> Result<Stmt, ParseError> {
        let loc = LineCol { line: self.peek_spanned().line, col: self.peek_spanned().column };
        self.advance(); // consume select
        self.expect(&Token::LBrace)?;

        let mut arms = Vec::new();
        while self.peek() != &Token::RBrace {
            self.expect(&Token::Pipe)?;

            let event = if self.peek() == &Token::Timeout {
                self.advance();
                self.expect(&Token::LParen)?;
                let ms = self.parse_expression()?;
                self.expect(&Token::RParen)?;
                SelectEvent::Timeout(ms)
            } else if self.peek() == &Token::Default {
                self.advance();
                SelectEvent::Default
            } else {
                let name = self.expect_ident()?;
                self.expect(&Token::Colon)?;
                let ty = self.parse_type()?;
                self.expect(&Token::ArrowLeft)?;
                let target = self.parse_expression()?;
                if matches!(&target, Expr::Await(_, _)) {
                    SelectEvent::Await { target, name, ty }
                } else {
                    SelectEvent::Recv { target, name, ty }
                }
            };

self.expect(&Token::FatArrow)?;
            let body = if self.peek() == &Token::LBrace {
                self.advance();
                let b = self.parse_block()?;
                self.expect(&Token::RBrace)?;
                b
            } else {
                let loc = LineCol {
                    line: self.peek_spanned().line,
                    col: self.peek_spanned().column,
                };
                let expr = self.parse_expression()?;
                vec![Stmt::Expression(loc, expr)]
            };
            arms.push(SelectArm { event, body });

            if self.peek() == &Token::Comma {
                self.advance();
            }
        }

        self.expect(&Token::RBrace)?;
        Ok(Stmt::Select { loc, arms })
    }

    fn parse_expression_statement(&mut self) -> Result<Stmt, ParseError> {
        let loc = LineCol { line: self.peek_spanned().line, col: self.peek_spanned().column };
        let expr = self.parse_expression()?;

        // Check if this is an assignment
        if self.peek() == &Token::Assign {
            self.advance();
            let value = self.parse_expression()?;
            self.expect(&Token::Semicolon)?;
            Ok(Stmt::Assignment {
                loc,
                target: expr,
                value,
            })
        } else {
            self.expect(&Token::Semicolon)?;
            Ok(Stmt::Expression(loc, expr))
        }
    }

    fn parse_expression(&mut self) -> Result<Expr, ParseError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_and()?;

        while self.peek() == &Token::PipePipe {
            self.advance();
            let right = self.parse_and()?;
            left = Expr::BinaryOp {
                span: Span::new(left.span().start, right.span().end),
                op: BinOp::Or,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_equality()?;

        while self.peek() == &Token::AmpAmp {
            self.advance();
            let right = self.parse_equality()?;
            left = Expr::BinaryOp {
                span: Span::new(left.span().start, right.span().end),
                op: BinOp::And,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_comparison()?;

        while self.peek() == &Token::Eq || self.peek() == &Token::Neq {
            let op = if self.peek() == &Token::Eq {
                self.advance();
                BinOp::Eq
            } else {
                self.advance();
                BinOp::Neq
            };
            let right = self.parse_comparison()?;
            left = Expr::BinaryOp {
                span: Span::new(left.span().start, right.span().end),
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_range()?;

        while matches!(self.peek(), Token::Lt | Token::Gt | Token::Le | Token::Ge) {
            let op = match self.peek() {
                Token::Lt => { self.advance(); BinOp::Lt }
                Token::Gt => { self.advance(); BinOp::Gt }
                Token::Le => { self.advance(); BinOp::Le }
                Token::Ge => { self.advance(); BinOp::Ge }
                _ => unreachable!(),
            };
            let right = self.parse_range()?;
            left = Expr::BinaryOp {
                span: Span::new(left.span().start, right.span().end),
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_range(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_concat()?;

        if self.peek() == &Token::DotDot || self.peek() == &Token::DotDotEq {
            let inclusive = self.peek() == &Token::DotDotEq;
            self.advance();
            let end = self.parse_concat()?;
            Ok(Expr::Range {
                span: Span::new(left.span().start, end.span().end),
                start: Box::new(left),
                end: Box::new(end),
                inclusive,
            })
        } else {
            Ok(left)
        }
    }

    fn parse_concat(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_additive()?;

        while self.peek() == &Token::Concat {
            self.advance();
            let right = self.parse_additive()?;
            left = Expr::BinaryOp {
                span: Span::new(left.span().start, right.span().end),
                op: BinOp::Concat,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_additive(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_multiplicative()?;

        while self.peek() == &Token::Plus || self.peek() == &Token::Minus {
            let op = if self.peek() == &Token::Plus {
                self.advance();
                BinOp::Add
            } else {
                self.advance();
                BinOp::Sub
            };
            let right = self.parse_multiplicative()?;
            left = Expr::BinaryOp {
                span: Span::new(left.span().start, right.span().end),
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_as()?;

        while matches!(self.peek(), Token::Star | Token::Slash | Token::Percent) {
            let op = match self.peek() {
                Token::Star => { self.advance(); BinOp::Mul }
                Token::Slash => { self.advance(); BinOp::Div }
                Token::Percent => { self.advance(); BinOp::Mod }
                _ => unreachable!(),
            };
            let right = self.parse_as()?;
            left = Expr::BinaryOp {
                span: Span::new(left.span().start, right.span().end),
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_as(&mut self) -> Result<Expr, ParseError> {
        let expr = self.parse_unary()?;

        if self.peek() == &Token::As {
            self.advance();
            let target_type = self.parse_type()?;
            Ok(Expr::Cast {
                span: Span::new(expr.span().start, self.last_end()),
                expr: Box::new(expr),
                target_type,
            })
        } else {
            Ok(expr)
        }
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        match self.peek() {
            Token::Minus => {
                let start = LineCol { line: self.peek_spanned().line, col: self.peek_spanned().column };
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::UnaryOp {
                    span: Span::new(start, expr.span().end),
                    op: UnaryOp::Neg,
                    expr: Box::new(expr),
                })
            }
            Token::Bang => {
                let start = LineCol { line: self.peek_spanned().line, col: self.peek_spanned().column };
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::UnaryOp {
                    span: Span::new(start, expr.span().end),
                    op: UnaryOp::Not,
                    expr: Box::new(expr),
                })
            }
            Token::Ref => {
                let start = LineCol { line: self.peek_spanned().line, col: self.peek_spanned().column };
                self.advance();
                let expr = self.parse_unary()?;
                let inner_span = expr.span();
                Ok(Expr::Ref(Box::new(expr), Span::new(start, inner_span.end)))
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;

        loop {
            match self.peek() {
                Token::Dot => {
                    self.advance();
                    let method = self.expect_ident()?;

                    if self.peek() == &Token::LParen {
                        self.advance();
                        let mut args = Vec::new();
                        if self.peek() != &Token::RParen {
                            loop {
                                args.push(self.parse_expression()?);
                                if self.peek() == &Token::Comma {
                                    self.advance();
                                } else {
                                    break;
                                }
                            }
                        }
                        self.expect(&Token::RParen)?;

                        expr = Expr::MethodCall {
                            span: Span::new(expr.span().start, self.last_end()),
                            object: Box::new(expr),
                            method,
                            args,
                        };
                    } else {
                        expr = Expr::FieldAccess {
                            span: Span::new(expr.span().start, self.last_end()),
                            object: Box::new(expr),
                            field: method,
                        };
                    }
                }
                Token::LParen => {
                    self.advance();
                    let mut args = Vec::new();
                    if self.peek() != &Token::RParen {
                        loop {
                            args.push(self.parse_expression()?);
                            if self.peek() == &Token::Comma {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(&Token::RParen)?;

                    expr = Expr::FunctionCall {
                        span: Span::new(expr.span().start, self.last_end()),
                        name: Box::new(expr),
                        args,
                    };
                }
                Token::LBracket => {
                    self.advance();
                    let index = self.parse_expression()?;
                    self.expect(&Token::RBracket)?;

                    expr = Expr::IndexAccess {
                        span: Span::new(expr.span().start, self.last_end()),
                        object: Box::new(expr),
                        index: Box::new(index),
                    };
                }
                Token::Await => {
                    self.advance();
                    let base_span = expr.span();
                    expr = Expr::Await(Box::new(expr), Span::new(base_span.start, self.last_end()));
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let spanned = self.peek_spanned().clone();

        match &spanned.token {
            Token::Integer(v) => {
                self.advance();
                Ok(Expr::IntegerLiteral(*v, self.span_of(&spanned)))
            }
            Token::Float(v) => {
                self.advance();
                Ok(Expr::FloatLiteral(*v, self.span_of(&spanned)))
            }
            Token::StringLiteral(s) => {
                self.advance();
                Ok(Expr::StringLiteral(s.clone(), self.span_of(&spanned)))
            }
            Token::True => {
                self.advance();
                Ok(Expr::BoolLiteral(true, self.span_of(&spanned)))
            }
            Token::False => {
                self.advance();
                Ok(Expr::BoolLiteral(false, self.span_of(&spanned)))
            }
            Token::Identifier(name) => {
                self.advance();
                // Check for qualified name: module::item
                if self.peek() == &Token::DoubleColon {
                    let mut path = vec![name.clone()];
                    while self.peek() == &Token::DoubleColon {
                        self.advance();
                        path.push(self.expect_ident()?);
                    }

                    // Check for enum constructor: Enum::Variant or Enum::Variant(args)
                    // Only uppercase-starting identifiers are enum type names
                    if path.len() == 2 && path[0].chars().next().map_or(false, |c| c.is_uppercase()) {
                        if self.peek() == &Token::LParen {
                            self.advance();
                            let mut args = Vec::new();
                            if self.peek() != &Token::RParen {
                                loop {
                                    args.push(self.parse_expression()?);
                                    if self.peek() == &Token::Comma {
                                        self.advance();
                                    } else {
                                        break;
                                    }
                                }
                            }
                            self.expect(&Token::RParen)?;
                            return Ok(Expr::EnumInit {
                                span: self.span_of(&spanned),
                                enum_name: path[0].clone(),
                                variant: path[1].clone(),
                                args,
                            });
                        } else {
                            // Enum variant without data: Shape::Point
                            return Ok(Expr::EnumInit {
                                span: self.span_of(&spanned),
                                enum_name: path[0].clone(),
                                variant: path[1].clone(),
                                args: vec![],
                            });
                        }
                    }

                    // Return as identifier with full path (e.g., "math::add")
                    let full_name = path.join("::");
                    Ok(Expr::Identifier(full_name, self.span_of(&spanned)))
                }
                // Check for struct initialization: TypeName { field: value, ... }
                // Only uppercase-starting identifiers can be struct type names
                else if name.chars().next().map_or(false, |c| c.is_uppercase()) && self.peek() == &Token::LBrace {
                    self.advance();
                    let mut fields = Vec::new();
                    while self.peek() != &Token::RBrace {
                        let field_name = self.expect_ident()?;
                        self.expect(&Token::Colon)?;
                        let field_value = self.parse_expression()?;
                        fields.push((field_name, field_value));
                        if self.peek() == &Token::Comma {
                            self.advance();
                        }
                    }
                    self.expect(&Token::RBrace)?;
                    Ok(Expr::StructInit {
                        span: self.span_of(&spanned),
                        name: name.clone(),
                        fields,
                    })
                } else {
                    Ok(Expr::Identifier(name.clone(), self.span_of(&spanned)))
                }
            }
            Token::LParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(&Token::RParen)?;
                Ok(expr)
            }
            Token::LBrace => {
                self.advance();
                let stmts = self.parse_block()?;
                self.expect(&Token::RBrace)?;
                Ok(Expr::Block(stmts, self.span_of(&spanned)))
            }
            Token::LBracket => {
                // Array literal [1, 2, 3]
                self.advance();
                let mut elements = Vec::new();
                if self.peek() != &Token::RBracket {
                    loop {
                        elements.push(self.parse_expression()?);
                        if self.peek() == &Token::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&Token::RBracket)?;
                Ok(Expr::ArrayLiteral(elements, self.span_of(&spanned)))
            }
            Token::Pound => {
                // Map literal: #{ "key": value, ... }
                self.advance();
                self.expect(&Token::LBrace)?;
                let mut pairs = Vec::new();
                if self.peek() != &Token::RBrace {
                    loop {
                        let key = self.parse_expression()?;
                        self.expect(&Token::Colon)?;
                        let value = self.parse_expression()?;
                        pairs.push((key, value));
                        if self.peek() == &Token::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&Token::RBrace)?;
                Ok(Expr::MapLiteral(pairs, self.span_of(&spanned)))
            }
            Token::If => self.parse_if(),
            Token::Match => self.parse_match(),
            Token::TypeResult => self.parse_builtin_enum_constructor("Result"),
            Token::TypeOption => self.parse_builtin_enum_constructor("Option"),
            Token::TypeChannel => self.parse_channel_constructor(),
            _ => Err(ParseError::UnexpectedToken(
                spanned.token.clone(),
                spanned.line,
                spanned.column,
                "expression".to_string(),
            )),
        }
    }

    /// Parse a built-in phantom enum constructor: Result::Ok(x), Option::Some(x), Option::None.
    fn parse_builtin_enum_constructor(
        &mut self,
        enum_name: &str,
    ) -> Result<Expr, ParseError> {
        let start = self.peek_spanned().clone();
        self.advance(); // consume Result / Option token
        self.expect(&Token::DoubleColon)?;
        let variant = self.expect_ident()?;

        let args = if self.peek() == &Token::LParen {
            self.advance();
            let mut args = Vec::new();
            if self.peek() != &Token::RParen {
                loop {
                    args.push(self.parse_expression()?);
                    if self.peek() == &Token::Comma {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            self.expect(&Token::RParen)?;
            args
        } else {
            vec![]
        };

        Ok(Expr::EnumInit {
            span: self.span_of(&start),
            enum_name: enum_name.to_string(),
            variant,
            args,
        })
    }

    /// Parse a channel constructor: Channel<T>(capacity).
    fn parse_channel_constructor(&mut self) -> Result<Expr, ParseError> {
        let start = self.peek_spanned().clone();
        self.advance(); // consume Channel token
        self.expect(&Token::Lt)?;
        let elem_type = self.parse_type()?;
        self.expect(&Token::Gt)?;
        self.expect(&Token::LParen)?;
        let capacity = self.parse_expression()?;
        self.expect(&Token::RParen)?;

        Ok(Expr::ChannelBounded {
            span: self.span_of(&start),
            elem_type: Box::new(elem_type),
            capacity: Box::new(capacity),
        })
    }

    fn parse_if(&mut self) -> Result<Expr, ParseError> {
        let start = self.peek_spanned().clone();
        self.advance(); // consume if
        let condition = self.parse_expression()?;
        let then_start = self.peek_spanned().clone();
        self.expect(&Token::LBrace)?;
        let then_stmts = self.parse_block()?;
        self.expect(&Token::RBrace)?;

        let else_branch = if self.peek() == &Token::Else {
            self.advance();
            if self.peek() == &Token::If {
                Some(Box::new(self.parse_if()?))
            } else {
                let else_start = self.peek_spanned().clone();
                self.expect(&Token::LBrace)?;
                let else_stmts = self.parse_block()?;
                self.expect(&Token::RBrace)?;
                Some(Box::new(Expr::Block(else_stmts, self.span_of(&else_start))))
            }
        } else {
            None
        };

        Ok(Expr::If {
            span: self.span_of(&start),
            condition: Box::new(condition),
            then_branch: Box::new(Expr::Block(then_stmts, self.span_of(&then_start))),
            else_branch,
        })
    }

    fn parse_match(&mut self) -> Result<Expr, ParseError> {
        let start = self.peek_spanned().clone();
        self.advance(); // consume match
        let expr = self.parse_expression()?;
        self.expect(&Token::LBrace)?;

        let mut arms = Vec::new();
        while self.peek() != &Token::RBrace {
            self.expect(&Token::Pipe)?;
            let pattern = self.parse_pattern()?;

            // Parse optional guard: if condition
            let guard = if self.peek() == &Token::If {
                self.advance();
                Some(self.parse_expression()?)
            } else {
                None
            };

            self.expect(&Token::FatArrow)?;
            let body = self.parse_expression()?;

            arms.push(MatchArm { pattern, guard, body });

            if self.peek() == &Token::Comma {
                self.advance();
            }
        }

        self.expect(&Token::RBrace)?;

        Ok(Expr::Match {
            span: self.span_of(&start),
            expr: Box::new(expr),
            arms,
        })
    }

    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        let spanned = self.peek_spanned().clone();

        match &spanned.token {
            Token::Integer(v) => {
                self.advance();
                Ok(Pattern::IntegerLiteral(*v))
            }
            Token::StringLiteral(s) => {
                self.advance();
                Ok(Pattern::StringLiteral(s.clone()))
            }
            Token::True => {
                self.advance();
                Ok(Pattern::BoolLiteral(true))
            }
            Token::False => {
                self.advance();
                Ok(Pattern::BoolLiteral(false))
            }
            Token::Underscore => {
                self.advance();
                Ok(Pattern::Wildcard)
            }
            Token::Identifier(name) => {
                self.advance();

                // Check if this is an enum variant: Enum::Variant or Enum::Variant(data)
                if self.peek() == &Token::DoubleColon {
                    self.advance();
                    let variant = self.expect_ident()?;

                    // Parse optional data: Variant(data1, data2)
                    let data = if self.peek() == &Token::LParen {
                        self.advance();
                        let mut data_patterns = Vec::new();
                        if self.peek() != &Token::RParen {
                            loop {
                                data_patterns.push(self.parse_pattern()?);
                                if self.peek() == &Token::Comma {
                                    self.advance();
                                } else {
                                    break;
                                }
                            }
                        }
                        self.expect(&Token::RParen)?;
                        Some(data_patterns)
                    } else {
                        None
                    };

                    return Ok(Pattern::EnumVariant {
                        enum_name: Some(name.clone()),
                        variant,
                        data,
                    });
                }

                // Check if this is Ok/Err/Some/None
                if matches!(name.as_str(), "Ok" | "Err" | "Some" | "None") {
                    if self.peek() == &Token::LParen {
                        self.advance();
                        let mut data_patterns = Vec::new();
                        if self.peek() != &Token::RParen {
                            loop {
                                data_patterns.push(self.parse_pattern()?);
                                if self.peek() == &Token::Comma {
                                    self.advance();
                                } else {
                                    break;
                                }
                            }
                        }
                        self.expect(&Token::RParen)?;
                        return Ok(Pattern::EnumVariant {
                            enum_name: None,
                            variant: name.clone(),
                            data: Some(data_patterns),
                        });
                    }
                    return Ok(Pattern::EnumVariant {
                        enum_name: None,
                        variant: name.clone(),
                        data: None,
                    });
                }

                Ok(Pattern::Identifier(name.clone()))
            }
            Token::Minus => {
                self.advance();
                let spanned = self.peek_spanned().clone();
                if let Token::Integer(v) = &spanned.token {
                    self.advance();
                    Ok(Pattern::IntegerLiteral(-v))
                } else {
                    Err(ParseError::UnexpectedToken(
                        spanned.token.clone(),
                        spanned.line,
                        spanned.column,
                        "integer literal".to_string(),
                    ))
                }
            }
            _ => Err(ParseError::UnexpectedToken(
                spanned.token.clone(),
                spanned.line,
                spanned.column,
                "pattern".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    #[test]
    fn test_parse_function() {
        let input = "@fn add(a: Int64, b: Int64) -> Int64 { return a + b; }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.items.len(), 1);
        match &program.items[0] {
            TopLevelItem::Function { name, params, return_type, .. } => {
                assert_eq!(name, "add");
                assert_eq!(params.len(), 2);
                assert_eq!(return_type, &Some(Type::Int64));
            }
            _ => panic!("Expected function"),
        }
    }

#[test]
    fn test_parse_struct() {
        let input = "@struct Point { x: Float64, y: Float64 }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.items.len(), 1);
        match &program.items[0] {
            TopLevelItem::Struct { name, fields, .. } => {
                assert_eq!(name, "Point");
                assert_eq!(fields.len(), 2);
            }
            _ => panic!("Expected struct"),
        }
    }

    #[test]
    fn test_parse_test_attribute() {
        let input = "@test @fn test_something() -> Void { }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.items.len(), 1);
        match &program.items[0] {
            TopLevelItem::Function {
                name,
                is_test,
                params,
                return_type,
                ..
            } => {
                assert_eq!(name, "test_something");
                assert!(is_test);
                assert_eq!(params.len(), 0);
                assert_eq!(return_type, &Some(Type::Void));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_parse_enum() {
        let input = "@enum Color { Red, Green, Blue }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        assert_eq!(program.items.len(), 1);
        match &program.items[0] {
            TopLevelItem::Enum { name, variants, .. } => {
                assert_eq!(name, "Color");
                assert_eq!(variants.len(), 3);
            }
            _ => panic!("Expected enum"),
        }
    }

    #[test]
    fn test_parse_while() {
        let input = "@fn count() -> Void { let i: Int64 = 0; while i < 10 { i = i + 1; } }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        match &program.items[0] {
            TopLevelItem::Function { body, .. } => {
                assert!(matches!(&body[1], Stmt::While { .. }));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_parse_cast() {
        let input = "@fn main() -> Void { let x: Int64 = 42 as Int64; }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        match &program.items[0] {
            TopLevelItem::Function { body, .. } => {
                if let Stmt::Let { value, .. } = &body[0] {
                    assert!(matches!(value, Expr::Cast { .. }));
                } else {
                    panic!("Expected let statement");
                }
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_parse_array_literal() {
        let input = "@fn main() -> Void { let arr: List<Int64> = [1, 2, 3]; }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        match &program.items[0] {
            TopLevelItem::Function { body, .. } => {
                if let Stmt::Let { value, .. } = &body[0] {
                    assert!(matches!(value, Expr::ArrayLiteral(_, _)));
                } else {
                    panic!("Expected let statement");
                }
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_parse_generic_function() {
        let input = "@fn identity<T>(x: T) -> T { return x; }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        match &program.items[0] {
            TopLevelItem::Function {
                name,
                type_params,
                params,
                return_type,
                ..
            } => {
                assert_eq!(name, "identity");
                assert_eq!(type_params, &vec!["T".to_string()]);
                assert_eq!(params.len(), 1);
                assert_eq!(params[0].ty, Type::Generic("T".to_string()));
                assert_eq!(return_type, &Some(Type::Generic("T".to_string())));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_parse_generic_nested_type() {
        let input = "@fn wrap<T>(x: T) -> Option<T> { return Option::Some(x); }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        match &program.items[0] {
            TopLevelItem::Function {
                type_params,
                return_type,
                ..
            } => {
                assert_eq!(type_params, &vec!["T".to_string()]);
                assert_eq!(
                    return_type,
                    &Some(Type::Option(Box::new(Type::Generic("T".to_string()))))
                );
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_parse_type_param_out_of_scope_is_custom() {
        let input = "@fn f(x: T) -> T { return x; }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        match &program.items[0] {
            TopLevelItem::Function {
                type_params,
                params,
                ..
            } => {
                assert!(type_params.is_empty());
                assert_eq!(params[0].ty, Type::Custom("T".to_string()));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_parse_channel_constructor() {
        let input = "@fn main() -> Void { let ch: Channel<Int64> = Channel<Int64>(10); }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        match &program.items[0] {
            TopLevelItem::Function { body, .. } => {
                if let Stmt::Let { ty, value, .. } = &body[0] {
                    assert_eq!(ty, &Some(Type::Channel(Box::new(Type::Int64))));
                    match value {
                        Expr::ChannelBounded { elem_type, .. } => {
                            assert_eq!(**elem_type, Type::Int64);
                        }
                        _ => panic!("Expected channel constructor"),
                    }
                } else {
                    panic!("Expected let statement");
                }
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_parse_async_function() {
        let input = "@fn async fetch(url: String) -> String { return url; }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        match &program.items[0] {
            TopLevelItem::Function { name, is_async, .. } => {
                assert_eq!(name, "fetch");
                assert!(is_async);
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_parse_async_impl_method() {
        let input = "@struct U { x: Int64 } @impl U { @fn async get(u: U) -> Int64 { return u.x; } }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        match &program.items[1] {
            TopLevelItem::Impl { methods, .. } => {
                assert_eq!(methods.len(), 1);
                assert!(methods[0].is_async);
            }
            _ => panic!("Expected impl"),
        }
    }

    #[test]
    fn test_parse_spawn_await() {
        let input = "@fn main() -> Void { spawn h; let s: String = h await; }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        match &program.items[0] {
            TopLevelItem::Function { body, .. } => {
                assert!(matches!(&body[0], Stmt::Spawn(..)));
                if let Stmt::Let { value, .. } = &body[1] {
                    assert!(matches!(value, Expr::Await(_, _)));
                } else {
                    panic!("Expected let statement");
                }
            }
            _ => panic!("Expected function"),
        }
    }
}
