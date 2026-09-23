use clap::{Parser as ClapParser, Subcommand};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::ast::*;
use crate::codegen::{compile_to_c, compile_to_c_tests};
use crate::lexer::Lexer;
use crate::modules::ModuleResolver;
use crate::parser::Parser;
use crate::typechecker::{TypeChecker, TypeError};

#[derive(ClapParser)]
#[command(name = "glyphc")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "Glyph language compiler", long_about = None)]
struct Cli {
    /// Start LSP server (backward compat for --lsp)
    #[arg(long, hide = true)]
    lsp: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a Glyph source file
    Compile {
        /// Input file path
        #[arg(short, long)]
        input: PathBuf,

        /// Output file path
        #[arg(short, long, default_value = "output.c")]
        output: PathBuf,

        /// Print LLVM IR to stdout
        #[arg(long)]
        emit_ir: bool,

        /// Skip type checking
        #[arg(long)]
        no_typecheck: bool,
    },

    /// Check syntax without generating code
    Check {
        /// Input file path
        #[arg(short, long)]
        input: PathBuf,
    },

    /// Print token stream
    Tokens {
        /// Input file path
        #[arg(short, long)]
        input: PathBuf,
    },

    /// Print AST
    Ast {
        /// Input file path
        #[arg(short, long)]
        input: PathBuf,
    },

    /// Compile and run a Glyph source file
    Run {
        /// Input file path
        #[arg(short, long)]
        input: PathBuf,

        /// Compiler to use (gcc or clang)
        #[arg(long, default_value = "gcc")]
        compiler: String,

        /// Optimization level (-O0, -O1, -O2, -O3)
        #[arg(long, default_value = "-O2")]
        opt: String,
    },

    /// Build a Glyph project using glyph.toml
    Build {
        /// Build profile (dev or release)
        #[arg(long, default_value = "dev")]
        profile: String,
    },

    /// Run @test functions
    Test {
        /// Input file path (if omitted, scans ./src for .glyph files)
        #[arg(short, long)]
        input: Option<PathBuf>,

        /// Compiler to use (gcc or clang)
        #[arg(long, default_value = "gcc")]
        compiler: String,

        /// Optimization level (-O0, -O1, -O2, -O3)
        #[arg(long, default_value = "-O2")]
        opt: String,
    },

    /// Format a Glyph source file (indentation/whitespace; comments preserved)
    Fmt {
        /// Input file path
        #[arg(short, long)]
        input: PathBuf,

        /// Report whether the file is already formatted (exit 1 if not)
        #[arg(long)]
        check: bool,

        /// Rewrite the file in place
        #[arg(long)]
        write: bool,
    },

    /// NeuralScript tensor compiler (port of nsc)
    Nns {
        /// Input .ns file
        input: PathBuf,

        /// Dump MLIR intermediate representation
        #[arg(long)]
        mlir: bool,

        /// Generate CPU C++ reference backend
        #[arg(
            long,
            help = "Generate CPU C++ backend (default is CUDA if neither --cpp nor --cuda is set)"
        )]
        cpp: bool,

        /// Select the CPU SIMD backend (currently a scalar reference alias)
        #[arg(long, conflicts_with = "cuda")]
        simd: bool,

        /// Generate CUDA backend (default)
        #[arg(
            long,
            help = "Generate CUDA backend (default if neither --cpp nor --cuda is set)"
        )]
        cuda: bool,

        /// Append the C-ABI runtime driver (ns_* host API)
        #[arg(long)]
        runtime: bool,

        /// Run static shape checking only
        #[arg(long)]
        check: bool,

        /// Enable fp16 (half-precision) codegen
        #[arg(long)]
        fp16: bool,

        /// Write output to file instead of stdout
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// LSP server (alias for `glyphc --lsp`)
    Lsp,

    /// Initialize a new Glyph project in the current directory
    Init {
        /// Project name (defaults to current directory name)
        name: Option<String>,
    },

    /// Create a new Glyph project in a new directory
    New {
        /// Name of the project / directory to create
        name: String,
    },
}

pub fn run() {
    let cli = Cli::parse();

    if cli.lsp {
        let mut server = crate::lsp::LspServer::new();
        server.run();
        return;
    }

    let Some(command) = cli.command else {
        eprintln!("No subcommand provided. Use --help.");
        process::exit(1);
    };

    match command {
        Commands::Compile {
            input,
            output,
            emit_ir,
            no_typecheck,
        } => {
            compile_file(&input, &output, emit_ir, no_typecheck);
        }
        Commands::Check { input } => {
            check_file(&input);
        }
        Commands::Tokens { input } => {
            print_tokens(&input);
        }
        Commands::Ast { input } => {
            print_ast(&input);
        }
        Commands::Run {
            input,
            compiler,
            opt,
        } => {
            run_file(&input, &compiler, &opt);
        }
        Commands::Build { profile } => {
            build_project(&profile);
        }
        Commands::Test {
            input,
            compiler,
            opt,
        } => {
            run_tests(input.as_deref(), &compiler, &opt);
        }
        Commands::Fmt {
            input,
            check,
            write,
        } => {
            fmt_file(&input, check, write);
        }
        Commands::Nns {
            input,
            mlir,
            cpp,
            simd,
            cuda,
            runtime,
            check,
            fp16,
            output,
        } => {
            run_nns(NnsRunOptions {
                input,
                mlir,
                cpp,
                simd,
                cuda,
                runtime,
                check,
                fp16,
                output,
            });
        }
        Commands::Lsp => {
            let mut server = crate::lsp::LspServer::new();
            server.run();
        }
        Commands::Init { name } => {
            init_project(name);
        }
        Commands::New { name } => {
            new_project(&name);
        }
    }
}

fn glyph_toml_content(name: &str) -> String {
    format!(
        r#"[package]
name = "{name}"
version = "0.1.0"
description = "A Glyph project"
authors = ["Your Name <you@example.com>"]

[build]
compiler = "gcc"
optimization = "-O2"
std = "glyph-1.0"
"#
    )
}

const HELLO_GLYPH: &str = r#"@fn main() -> Void {
    print("Hello, world!");
}
"#;

fn sanitize_name(raw: &str) -> String {
    // Glyph project name should be valid id: alphanumeric + _ -
    // Replace invalid chars with _ and trim.
    let sanitized: String = raw
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "my-project".to_string()
    } else {
        sanitized
    }
}

fn write_project_scaffold(project_dir: &Path, name: &str) {
    let glyph_toml_path = project_dir.join("glyph.toml");
    if glyph_toml_path.exists() {
        eprintln!(
            "Error: glyph.toml already exists at {}",
            glyph_toml_path.display()
        );
        process::exit(1);
    }

    let src_dir = project_dir.join("src");
    fs::create_dir_all(&src_dir).unwrap_or_else(|e| {
        eprintln!("Error creating {}: {}", src_dir.display(), e);
        process::exit(1);
    });

    let toml_content = glyph_toml_content(name);
    fs::write(&glyph_toml_path, toml_content).unwrap_or_else(|e| {
        eprintln!("Error writing {}: {}", glyph_toml_path.display(), e);
        process::exit(1);
    });

    let main_path = src_dir.join("main.glyph");
    if !main_path.exists() {
        fs::write(&main_path, HELLO_GLYPH).unwrap_or_else(|e| {
            eprintln!("Error writing {}: {}", main_path.display(), e);
            process::exit(1);
        });
    }

    println!("Created project '{}' at {}", name, project_dir.display());
    println!("  glyph.toml");
    println!("  src/main.glyph");
    println!("\nNext steps:");
    println!("  glyphc build          # build the project");
    println!("  glyphc run --input src/main.glyph");
}

fn init_project(name: Option<String>) {
    let cwd = std::env::current_dir().unwrap_or_else(|e| {
        eprintln!("Error getting current directory: {}", e);
        process::exit(1);
    });

    let project_name = match name {
        Some(n) => sanitize_name(&n),
        None => {
            let dir_name = cwd
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("my-project");
            sanitize_name(dir_name)
        }
    };

    write_project_scaffold(&cwd, &project_name);
}

fn new_project(name: &str) {
    // Keep the requested path intact (including absolute paths and nested
    // directories); only the manifest name is sanitized. Sanitizing the whole
    // path would turn `/tmp/demo` into `_tmp_demo` in the current directory.
    let project_dir = PathBuf::from(name);
    let project_name = project_dir
        .file_name()
        .and_then(|n| n.to_str())
        .map(sanitize_name)
        .unwrap_or_else(|| "my-project".to_string());

    if project_dir.exists() {
        eprintln!(
            "Error: directory '{}' already exists",
            project_dir.display()
        );
        process::exit(1);
    }

    fs::create_dir_all(&project_dir).unwrap_or_else(|e| {
        eprintln!("Error creating directory {}: {}", project_dir.display(), e);
        process::exit(1);
    });

    write_project_scaffold(&project_dir, &project_name);
}

fn fmt_file(input: &Path, check: bool, write: bool) {
    let source = read_source(input);

    // Never reformat code the compiler cannot lex/parse: validation first.
    let tokens = match Lexer::new(&source).tokenize() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("syntax error: {}", e);
            process::exit(1);
        }
    };
    if let Err(e) = Parser::new(tokens).parse_program() {
        eprintln!("syntax error: {}", e);
        process::exit(1);
    }

    let formatted = match crate::formatter::format(&source) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("format error: {}", e);
            process::exit(1);
        }
    };

    if check {
        if formatted == source {
            println!("{}: already formatted", input.display());
            return;
        }
        eprintln!(
            "{}: not formatted (whitespace/indentation differs; run 'glyphc fmt --input ... --write')",
            input.display()
        );
        if write {
            // fall through to write below
        } else {
            process::exit(1);
        }
    }

    if write {
        if let Err(e) = fs::write(input, &formatted) {
            eprintln!("Error writing {}: {}", input.display(), e);
            process::exit(1);
        }
        println!("formatted {}", input.display());
    } else if !check {
        print!("{}", formatted);
    }
}

fn read_source(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading file {}: {}", path.display(), e);
        process::exit(1);
    })
}

fn compile_file(input: &Path, output: &Path, emit_ir: bool, no_typecheck: bool) {
    if let Err(e) = compile_file_opts(input, output, emit_ir, no_typecheck, None, false) {
        eprintln!("{}", e);
        process::exit(1);
    }
}

fn compile_file_opts(
    input: &Path,
    output: &Path,
    emit_ir: bool,
    no_typecheck: bool,
    test_names: Option<&[String]>,
    quiet: bool,
) -> Result<(), String> {
    // Resolve modules
    let root_dir = input
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let mut resolver = ModuleResolver::new(root_dir.clone());

    if let Err(e) = resolver.resolve(input) {
        return Err(format!("Module error: {}", e));
    }

    // Get topological order
    let sorted = match resolver.topological_sort() {
        Ok(order) => order,
        Err(e) => return Err(format!("Module error: {}", e)),
    };

    // Merge all module ASTs into one Program, prefixing names with module name
    let input_module_name = resolver.path_to_module_name(input);
    let mut all_items = Vec::new();
    for module_name in &sorted {
        if let Some(module) = resolver.get_module(module_name) {
            // Don't prefix the entry point module
            let prefix = if module_name == &input_module_name {
                String::new()
            } else {
                format!("{}_", module_name.replace("::", "_"))
            };

            // Collect names this module declares itself (its own functions/consts),
            // so unqualified references inside its bodies can be reprefixed too.
            let mut self_names = HashSet::new();
            for item in &module.ast.items {
                match item {
                    TopLevelItem::Function { name, .. } => {
                        self_names.insert(name.clone());
                    }
                    TopLevelItem::Const { name, .. } => {
                        self_names.insert(name.clone());
                    }
                    _ => {}
                }
            }

            for item in &module.ast.items {
                let mut prefixed = prefix_item(item, &prefix);
                if !prefix.is_empty() {
                    rewrite_self_references(&mut prefixed, &self_names, &prefix);
                }
                all_items.push(prefixed);
            }
        }
    }

    let merged_program = Program { items: all_items };

    // Type checking
    if !no_typecheck {
        let mut checker = TypeChecker::new();
        if let Err(e) = checker.check_program(&merged_program) {
            return Err(format!("Type error: {}", e));
        }
    }

    // Code generation
    if emit_ir {
        println!("// C code for {}", input.display());
        println!("// ---");
    }

    let gen = if let Some(tests) = test_names {
        compile_to_c_tests(&merged_program, &output.to_string_lossy(), tests)
    } else {
        compile_to_c(&merged_program, &output.to_string_lossy())
    };

    match gen {
        Ok(()) => {
            if emit_ir {
                println!("// C code written to {}", output.display());
            } else if !quiet {
                println!("Compiled successfully to {}", output.display());
            }
            Ok(())
        }
        Err(e) => Err(format!("Code generation error: {}", e)),
    }
}

fn prefix_item(item: &TopLevelItem, prefix: &str) -> TopLevelItem {
    match item {
        TopLevelItem::Function {
            name,
            name_span,
            type_params,
            params,
            return_type,
            body,
            is_async,
            is_test,
            pub_vis,
        } => {
            // Don't prefix main function - it's the entry point
            let new_name = if name == "main" {
                name.clone()
            } else {
                format!("{}{}", prefix, name)
            };
            TopLevelItem::Function {
                name: new_name,
                name_span: *name_span,
                type_params: type_params.clone(),
                params: params.clone(),
                return_type: return_type.clone(),
                body: body.clone(),
                is_async: *is_async,
                is_test: *is_test,
                pub_vis: *pub_vis,
            }
        }
        TopLevelItem::Struct {
            name,
            name_span,
            fields,
            pub_vis,
        } => TopLevelItem::Struct {
            name: format!("{}{}", prefix, name),
            name_span: *name_span,
            fields: fields.clone(),
            pub_vis: *pub_vis,
        },
        TopLevelItem::Enum {
            name,
            variants,
            pub_vis,
        } => TopLevelItem::Enum {
            name: format!("{}{}", prefix, name),
            variants: variants.clone(),
            pub_vis: *pub_vis,
        },
        TopLevelItem::Const {
            name,
            ty,
            value,
            pub_vis,
        } => TopLevelItem::Const {
            name: format!("{}{}", prefix, name),
            ty: ty.clone(),
            value: value.clone(),
            pub_vis: *pub_vis,
        },
        TopLevelItem::Impl {
            type_name,
            methods,
            pub_vis,
        } => TopLevelItem::Impl {
            type_name: format!("{}{}", prefix, type_name),
            methods: methods.clone(),
            pub_vis: *pub_vis,
        },
        other => other.clone(),
    }
}

/// Rewrite unqualified references to the module's own top-level names
/// (functions/consts) inside its item bodies after prefixing.
fn rewrite_self_references(item: &mut TopLevelItem, self_names: &HashSet<String>, prefix: &str) {
    match item {
        TopLevelItem::Function { body, .. } => {
            for stmt in body {
                remap_stmt(stmt, self_names, prefix);
            }
        }
        TopLevelItem::Const { value, .. } => {
            remap_expr(value, self_names, prefix);
        }
        TopLevelItem::Impl { methods, .. } => {
            for method in methods {
                for stmt in &mut method.body {
                    remap_stmt(stmt, self_names, prefix);
                }
            }
        }
        _ => {}
    }
}

fn remap_stmt(stmt: &mut Stmt, self_names: &HashSet<String>, prefix: &str) {
    match stmt {
        Stmt::Let { value, .. } => remap_expr(value, self_names, prefix),
        Stmt::Assignment { target, value, .. } => {
            remap_expr(target, self_names, prefix);
            remap_expr(value, self_names, prefix);
        }
        Stmt::Expression(_, expr) => remap_expr(expr, self_names, prefix),
        Stmt::Return(_, Some(expr)) => remap_expr(expr, self_names, prefix),
        Stmt::Loop(_, body) => {
            for s in body {
                remap_stmt(s, self_names, prefix);
            }
        }
        Stmt::While {
            condition, body, ..
        } => {
            remap_expr(condition, self_names, prefix);
            for s in body {
                remap_stmt(s, self_names, prefix);
            }
        }
        Stmt::For { iterable, body, .. } => {
            remap_expr(iterable, self_names, prefix);
            for s in body {
                remap_stmt(s, self_names, prefix);
            }
        }
        Stmt::Guard {
            condition,
            else_body,
            ..
        } => {
            remap_expr(condition, self_names, prefix);
            for s in else_body {
                remap_stmt(s, self_names, prefix);
            }
        }
        Stmt::Spawn(_, expr) => remap_expr(expr, self_names, prefix),
        _ => {}
    }
}

fn remap_expr(expr: &mut Expr, self_names: &HashSet<String>, prefix: &str) {
    match expr {
        Expr::IntegerLiteral(_, _)
        | Expr::FloatLiteral(_, _)
        | Expr::StringLiteral(_, _)
        | Expr::BoolLiteral(_, _) => {}
        Expr::Identifier(name, _) => {
            if self_names.contains(name) {
                *name = format!("{}{}", prefix, name);
            }
        }
        Expr::BinaryOp { left, right, .. } => {
            remap_expr(left, self_names, prefix);
            remap_expr(right, self_names, prefix);
        }
        Expr::UnaryOp { expr: inner, .. } => remap_expr(inner, self_names, prefix),
        Expr::Ref(inner, _) => remap_expr(inner, self_names, prefix),
        Expr::Cast { expr: inner, .. } => remap_expr(inner, self_names, prefix),
        Expr::FunctionCall { name, args, .. } => {
            remap_expr(name, self_names, prefix);
            for arg in args {
                remap_expr(arg, self_names, prefix);
            }
        }
        Expr::MethodCall { object, args, .. } => {
            remap_expr(object, self_names, prefix);
            for arg in args {
                remap_expr(arg, self_names, prefix);
            }
        }
        Expr::FieldAccess { object, .. } => remap_expr(object, self_names, prefix),
        Expr::IndexAccess { object, index, .. } => {
            remap_expr(object, self_names, prefix);
            remap_expr(index, self_names, prefix);
        }
        Expr::StructInit { fields, .. } => {
            for (_, value) in fields {
                remap_expr(value, self_names, prefix);
            }
        }
        Expr::EnumInit { args, .. } => {
            for arg in args {
                remap_expr(arg, self_names, prefix);
            }
        }
        Expr::Match {
            expr: inner, arms, ..
        } => {
            remap_expr(inner, self_names, prefix);
            for arm in arms {
                if let Some(guard) = &mut arm.guard {
                    remap_expr(guard, self_names, prefix);
                }
                remap_expr(&mut arm.body, self_names, prefix);
            }
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            remap_expr(condition, self_names, prefix);
            remap_expr(then_branch, self_names, prefix);
            if let Some(else_branch) = else_branch {
                remap_expr(else_branch, self_names, prefix);
            }
        }
        Expr::Block(stmts, _) => {
            for stmt in stmts {
                remap_stmt(stmt, self_names, prefix);
            }
        }
        Expr::Range { start, end, .. } => {
            remap_expr(start, self_names, prefix);
            remap_expr(end, self_names, prefix);
        }
        Expr::ArrayLiteral(elements, _) => {
            for elem in elements {
                remap_expr(elem, self_names, prefix);
            }
        }
        Expr::MapLiteral(pairs, _) => {
            for (k, v) in pairs {
                remap_expr(k, self_names, prefix);
                remap_expr(v, self_names, prefix);
            }
        }
        Expr::ChannelBounded { capacity, .. } => remap_expr(capacity, self_names, prefix),
        Expr::Await(inner, _) => remap_expr(inner, self_names, prefix),
    }
}

fn type_error_span(e: &TypeError) -> Option<LineCol> {
    match e {
        TypeError::InFunction { source, .. } => type_error_span(source),
        TypeError::AtLine { loc, .. } => Some(*loc),
        TypeError::AtSpan { span, .. } => Some(span.start),
        _ => None,
    }
}

/// Prints a source snippet with a caret pointing at the failing statement.
fn print_type_error_snippet(source: &str, e: &TypeError) {
    let Some(loc) = type_error_span(e) else {
        return;
    };
    let Some(raw) = source.lines().nth(loc.line.saturating_sub(1)) else {
        return;
    };
    let mut text: String = raw.chars().take(100).collect();
    let truncated = raw.chars().count() > 100;
    if truncated {
        text.push('…');
    }
    eprintln!("  --> {}:{}", loc.line, loc.col);
    eprintln!("   |");
    eprintln!(" {:>3} | {}", loc.line, text);
    eprintln!("     | {}^", " ".repeat(loc.col.saturating_sub(1)));
}

fn check_file(input: &Path) {
    let source = read_source(input);

    // Lexing
    let mut lexer = Lexer::new(&source);
    let tokens = match lexer.tokenize() {
        Ok(tokens) => tokens,
        Err(e) => {
            eprintln!("Lexer error: {}", e);
            process::exit(1);
        }
    };

    // Parsing
    let mut parser = Parser::new(tokens);
    let program = match parser.parse_program() {
        Ok(program) => program,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            process::exit(1);
        }
    };

    // Type checking
    let mut checker = TypeChecker::new();
    match checker.check_program(&program) {
        Ok(()) => {
            println!("✓ File {} is valid", input.display());
        }
        Err(e) => {
            print_type_error_snippet(&source, &e);
            eprintln!("Type error: {}", e);
            process::exit(1);
        }
    }
}

fn print_tokens(input: &Path) {
    let source = read_source(input);
    let mut lexer = Lexer::new(&source);

    match lexer.tokenize() {
        Ok(tokens) => {
            for token in &tokens {
                println!("[{:>3}:{:>3}] {:?}", token.line, token.column, token.token);
            }
        }
        Err(e) => {
            eprintln!("Lexer error: {}", e);
            process::exit(1);
        }
    }
}

fn print_ast(input: &Path) {
    let source = read_source(input);

    // Lexing
    let mut lexer = Lexer::new(&source);
    let tokens = match lexer.tokenize() {
        Ok(tokens) => tokens,
        Err(e) => {
            eprintln!("Lexer error: {}", e);
            process::exit(1);
        }
    };

    // Parsing
    let mut parser = Parser::new(tokens);
    match parser.parse_program() {
        Ok(program) => {
            println!("{:#?}", program);
        }
        Err(e) => {
            eprintln!("Parse error: {}", e);
            process::exit(1);
        }
    }
}

fn make_temp_dir(kind: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("glyphc-{kind}-{}-{stamp}", process::id()));
    fs::create_dir(&path).unwrap_or_else(|e| {
        eprintln!("Error creating temp dir {}: {}", path.display(), e);
        process::exit(1);
    });
    path
}

fn run_file(input: &Path, compiler: &str, opt: &str) {
    let temp_dir = make_temp_dir("run");

    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("glyph_output")
        .to_string();
    let c_file = temp_dir.join(format!("{}.c", stem));
    let binary = temp_dir.join(&stem);

    // Compile Glyph to C
    compile_file(input, &c_file, false, false);

    // Compile C to binary
    let output = process::Command::new(compiler)
        .args([
            opt,
            "-std=gnu11",
            "-pthread",
            "-o",
            binary.to_string_lossy().as_ref(),
            c_file.to_string_lossy().as_ref(),
            "-lm",
        ])
        .output()
        .unwrap_or_else(|e| {
            eprintln!("Error running {}: {}", compiler, e);
            process::exit(1);
        });

    if !output.status.success() {
        eprintln!(
            "Compilation error:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        process::exit(1);
    }

    // Run the binary, inheriting stdio so interactive programs (read_line)
    // and their output behave normally.
    let status = process::Command::new(&binary).status().unwrap_or_else(|e| {
        eprintln!("Error running binary: {}", e);
        process::exit(1);
    });

    if !status.success() {
        process::exit(status.code().unwrap_or(1));
    }

    // Cleanup
    let _ = fs::remove_dir_all(&temp_dir);
}

fn run_tests(input: Option<&Path>, compiler: &str, opt: &str) {
    let temp_dir = make_temp_dir("test");

    // Collect candidate .glyph files
    let mut files: Vec<PathBuf> = Vec::new();
    if let Some(path) = input {
        files.push(path.to_path_buf());
    } else {
        let src_dir = PathBuf::from("src");
        if !src_dir.exists() {
            eprintln!("Error: no --input given and 'src' directory not found");
            process::exit(1);
        }
        collect_glyph_files(&src_dir, &mut files);
        files.sort();
    }

    if files.is_empty() {
        eprintln!("No .glyph files found");
        process::exit(1);
    }

    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let yellow = "\x1b[33m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";

    let mut total_files = 0usize;
    let mut total_passed = 0usize;
    let mut total_failed = 0usize;
    let mut any_failure = false;

    for (file_index, file) in files.iter().enumerate() {
        // Only target files that declare @test functions
        let test_names = match extract_test_functions(file) {
            Ok(names) if !names.is_empty() => names,
            Ok(_) => continue, // no tests in this file
            Err(e) => {
                println!("{}[ERROR]{} {}: {}", red, reset, file.display(), e);
                any_failure = true;
                continue;
            }
        };

        let stem = file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("glyph_test")
            .to_string();
        let c_file = temp_dir.join(format!("{file_index}-{stem}.test.c"));
        let binary = temp_dir.join(format!("{file_index}-{stem}.test"));

        println!(
            "{}Running {} test(s) in {}...{}",
            yellow,
            test_names.len(),
            file.display(),
            reset
        );

        // Glyph -> C (test mode): errors handled per-file
        if let Err(e) = compile_file_opts(file, &c_file, false, false, Some(&test_names), true) {
            println!("  {}{}{}", red, e, reset);
            any_failure = true;
            continue;
        }

        // C -> binary
        let compile = process::Command::new(compiler)
            .args([
                opt,
                "-std=gnu11",
                "-pthread",
                "-o",
                binary.to_string_lossy().as_ref(),
                c_file.to_string_lossy().as_ref(),
                "-lm",
            ])
            .output()
            .unwrap_or_else(|e| {
                eprintln!("Error running {}: {}", compiler, e);
                process::exit(1);
            });

        if !compile.status.success() {
            println!("  {}Compilation error:{}", red, reset);
            for line in String::from_utf8_lossy(&compile.stderr).lines() {
                println!("    {}", line);
            }
            any_failure = true;
            continue;
        }

        // Run the test binary and parse the report
        let run = process::Command::new(&binary).output().unwrap_or_else(|e| {
            eprintln!("Error running test binary {}: {}", binary.display(), e);
            process::exit(1);
        });

        let stdout = String::from_utf8_lossy(&run.stdout).to_string();
        let stderr = String::from_utf8_lossy(&run.stderr).to_string();
        let mut body_lines: Vec<String> = Vec::new();
        let mut file_passed = 0usize;
        let mut file_failed = 0usize;

        for line in stdout.lines() {
            if let Some(name) = line.strip_prefix("[test] ") {
                if name.ends_with(" ok") {
                    println!("  {}[ok]{} {}", green, reset, name.trim_end_matches(" ok"));
                    file_passed += 1;
                } else if name.ends_with(" FAILED") {
                    println!(
                        "  {}[FAILED]{} {}",
                        red,
                        reset,
                        name.trim_end_matches(" FAILED")
                    );
                    file_failed += 1;
                }
            } else if line.starts_with("[summary]") {
                // machine-read report; pass/fail already counted above
            } else if !line.is_empty() {
                body_lines.push(line.to_string());
            }
        }

        if !stderr.trim().is_empty() {
            body_lines.push(stderr.trim().to_string());
        }

        // For failed binaries, surface captured output / assertion messages
        if file_failed > 0 || (run.status.code() != Some(0)) {
            for line in &body_lines {
                println!("    {}", line);
            }
        }

        total_passed += file_passed;
        total_failed += file_failed;
        total_files += 1;
        if file_failed > 0 || (run.status.code() != Some(0)) {
            any_failure = true;
        }

        let _ = fs::remove_file(&c_file);
        let _ = fs::remove_file(&binary);
    }

    // Summary
    println!();
    let status = if any_failure {
        format!("{}{}FAILED{}", bold, red, reset)
    } else {
        format!("{}{}OK{}", bold, green, reset)
    };
    println!(
        "{}Result:{} {}  ({} file(s), {} passed, {} failed)",
        bold, reset, status, total_files, total_passed, total_failed
    );

    let _ = fs::remove_dir_all(&temp_dir);
    if any_failure {
        process::exit(1);
    }
}

fn collect_glyph_files(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_glyph_files(&path, out);
            } else if path.extension().is_some_and(|e| e == "glyph") {
                out.push(path);
            }
        }
    }
}

/// Parse a file and return the names of its @test functions (in declaration order).
fn extract_test_functions(path: &Path) -> Result<Vec<String>, String> {
    let source = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut lexer = Lexer::new(&source);
    let tokens = lexer
        .tokenize()
        .map_err(|e| format!("Lexer error: {}", e))?;
    let mut parser = Parser::new(tokens);
    let program = parser
        .parse_program()
        .map_err(|e| format!("Parse error: {}", e))?;

    Ok(program
        .items
        .iter()
        .filter_map(|item| match item {
            TopLevelItem::Function {
                name,
                is_test: true,
                ..
            } => Some(name.clone()),
            _ => None,
        })
        .collect())
}

fn build_project(profile: &str) {
    // Read glyph.toml
    let manifest_path = PathBuf::from("glyph.toml");
    if !manifest_path.exists() {
        eprintln!("Error: glyph.toml not found in current directory");
        process::exit(1);
    }

    let manifest_content = fs::read_to_string(&manifest_path).unwrap_or_else(|e| {
        eprintln!("Error reading glyph.toml: {}", e);
        process::exit(1);
    });

    // Parse manifest (simple TOML parsing)
    let mut compiler = "gcc".to_string();
    let mut optimization = "-O2".to_string();

    for line in manifest_content.lines() {
        let line = line.trim();
        if line.starts_with("compiler") {
            if let Some(value) = line.split('=').nth(1) {
                compiler = value.trim().trim_matches('"').to_string();
            }
        } else if line.starts_with("optimization") {
            if let Some(value) = line.split('=').nth(1) {
                optimization = value.trim().trim_matches('"').to_string();
            }
        }
    }

    // Override optimization for release profile
    if profile == "release" {
        optimization = "-O3".to_string();
    }

    println!("Building project with profile: {}", profile);
    println!("Compiler: {}", compiler);
    println!("Optimization: {}", optimization);

    // Find all .glyph files in src directory
    let src_dir = PathBuf::from("src");
    if !src_dir.exists() {
        eprintln!("Error: src directory not found");
        process::exit(1);
    }

    let mut glyph_files = Vec::new();
    for entry in fs::read_dir(&src_dir).unwrap_or_else(|e| {
        eprintln!("Error reading src directory: {}", e);
        process::exit(1);
    }) {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Warning: skipping unreadable entry: {}", e);
                continue;
            }
        };
        let path = entry.path();
        if path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default()
            == "glyph"
        {
            glyph_files.push(path);
        }
    }

    if glyph_files.is_empty() {
        eprintln!("Error: no .glyph files found in src directory");
        process::exit(1);
    }

    println!("Found {} glyph files", glyph_files.len());

    // Create build directory
    let build_dir = PathBuf::from("build");
    fs::create_dir_all(&build_dir).unwrap_or_else(|e| {
        eprintln!("Error creating build directory: {}", e);
        process::exit(1);
    });

    // Compile each file
    let mut c_files = Vec::new();
    for glyph_file in &glyph_files {
        let stem = glyph_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("glyph_module")
            .to_string();
        let c_file = build_dir.join(format!("{}.c", stem));

        println!("Compiling {} -> {}", glyph_file.display(), c_file.display());
        compile_file(glyph_file, &c_file, false, false);
        c_files.push(c_file);
    }

    // Link all C files
    let output_name = "main";
    let output_path = build_dir.join(output_name);

    let mut args = vec![
        optimization.as_str().to_string(),
        "-std=gnu11".to_string(),
        "-pthread".to_string(),
        "-o".to_string(),
        output_path.to_string_lossy().to_string(),
    ];
    for c_file in &c_files {
        args.push(c_file.to_string_lossy().to_string());
    }
    args.push("-lm".to_string());

    println!("Linking...");

    let output = process::Command::new(&compiler)
        .args(&args)
        .output()
        .unwrap_or_else(|e| {
            eprintln!("Error running {}: {}", compiler, e);
            process::exit(1);
        });

    if !output.status.success() {
        eprintln!(
            "Linking error:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        process::exit(1);
    }

    println!("Build successful: {}", output_path.display());
}

fn nns_parse_ns_error(msg: &str) -> (u32, u32, String) {
    // Try to extract " at <line>:<col>" with optional "line " prefix.
    // Returns (line, col, message) where message is the part after the location,
    // or prefix before " at " if no suffix exists. Falls back to 1:1.
    if let Some(pos) = msg.rfind(" at ") {
        let after_raw = &msg[pos + 4..];
        let after = after_raw.strip_prefix("line ").unwrap_or(after_raw);
        // parse line
        let mut idx = 0usize;
        let bytes = after.as_bytes();
        while idx < bytes.len() && bytes[idx].is_ascii_digit() {
            idx += 1;
        }
        if idx > 0 {
            if let Ok(line) = after[..idx].parse::<u32>() {
                if idx < bytes.len() && bytes[idx] == b':' {
                    let mut j = idx + 1;
                    let col_start = j;
                    while j < bytes.len() && bytes[j].is_ascii_digit() {
                        j += 1;
                    }
                    if j > col_start {
                        if let Ok(col) = after[col_start..j].parse::<u32>() {
                            let rest = &after[j..];
                            let rest_trim = rest.trim_start_matches([':', ' ']).trim();
                            let prefix = msg[..pos].trim();
                            // Decide message: for "Expected ... at X:Y: suffix", combine prefix + suffix
                            // for "Parse error at X:Y: msg", use suffix only
                            let message = if !rest_trim.is_empty() {
                                if prefix == "Parse error" || prefix.is_empty() {
                                    rest_trim.to_string()
                                } else if prefix.starts_with("Expected") {
                                    format!("{}: {}", prefix, rest_trim)
                                } else if prefix.starts_with("Unexpected") && rest_trim.is_empty() {
                                    prefix.to_string()
                                } else if !rest_trim.is_empty()
                                    && !prefix.is_empty()
                                    && !prefix.starts_with("Parse error")
                                {
                                    // generic: if suffix is non-empty use suffix (e.g. "Unexpected token...")
                                    rest_trim.to_string()
                                } else {
                                    rest_trim.to_string()
                                }
                            } else {
                                if !prefix.is_empty() {
                                    prefix.to_string()
                                } else {
                                    msg.to_string()
                                }
                            };
                            return (line, col, message);
                        }
                    }
                } else {
                    // only line, no col
                    let rest = &after[idx..];
                    let rest_trim = rest.trim_start_matches([':', ' ']).trim();
                    let prefix = msg[..pos].trim();
                    let message = if !rest_trim.is_empty() {
                        rest_trim.to_string()
                    } else if !prefix.is_empty() {
                        prefix.to_string()
                    } else {
                        msg.to_string()
                    };
                    return (line, 1, message);
                }
            }
        }
    }
    // No location found -> 1:1 with original message
    // Strip leading "Parse error at ..." fallback if still present (should have been handled)
    (1, 1, msg.to_string())
}

struct NnsRunOptions {
    input: PathBuf,
    mlir: bool,
    cpp: bool,
    simd: bool,
    cuda: bool,
    runtime: bool,
    check: bool,
    fp16: bool,
    output: Option<PathBuf>,
}

fn run_nns(options: NnsRunOptions) {
    let NnsRunOptions {
        input,
        mlir,
        cpp,
        simd,
        cuda,
        runtime,
        check,
        fp16,
        output,
    } = options;

    // Read source file, mirroring nsc's "Cannot open file: <path>"
    let source = match fs::read_to_string(&input) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("Error: Cannot open file: {}", input.display());
            process::exit(1);
        }
    };

    // Stage 1: Lex
    let mut lexer = crate::nns::lexer::Lexer::new(&source);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(e) => {
            let (line, col, msg) = nns_parse_ns_error(&e.0);
            eprintln!("Static shape/type errors:");
            eprintln!("  {}:{}  {}", line, col, msg);
            process::exit(1);
        }
    };

    // Stage 2: Parse
    let mut parser = crate::nns::parser::Parser::new(tokens);
    let mut program = match parser.parse_program() {
        Ok(p) => p,
        Err(e) => {
            let (line, col, msg) = nns_parse_ns_error(&e.0);
            eprintln!("Static shape/type errors:");
            eprintln!("  {}:{}  {}", line, col, msg);
            process::exit(1);
        }
    };

    // Stage 3: Type + shape checking
    let mut checker = crate::nns::shape_checker::ShapeChecker::new();
    if !checker.check(&mut program) {
        eprintln!("Static shape/type errors:");
        for e in checker.errors() {
            let line = if e.line == 0 { 1 } else { e.line };
            let col = if e.column == 0 { 1 } else { e.column };
            eprintln!("  {}:{}  {}", line, col, e.message);
        }
        if checker.errors().is_empty() {
            eprintln!("  1:1  shape checking failed");
        }
        process::exit(1);
    }

    if check {
        println!("Shape checking passed.");
        return;
    }

    // Stage 4: MLIR lowering
    let mut mlir_compiler = crate::nns::mlir::mlir_compiler::MLIRCompiler::new();
    let mut module = mlir_compiler.compile(&mut program);

    if mlir {
        let dump = crate::nns::mlir::mlir_compiler::MLIRCompiler::dump(&module);
        if let Some(out_path) = output {
            if let Err(e) = fs::write(&out_path, &dump) {
                eprintln!("Error writing {}: {}", out_path.display(), e);
                process::exit(1);
            }
        } else {
            print!("{}", dump);
        }
        return;
    }

    // Stage 4b: kernel fusion pass (matmul + activation/layernorm/bias).
    let mut fuse = crate::nns::mlir::fusion::FusionPass::new();
    let nfused = fuse.run(&mut module);
    if !check && !mlir {
        eprintln!("Fusion: {} groups fused", nfused);
    }

    // Stage 5: Codegen
    let backend = if simd && !cuda {
        crate::nns::codegen::codegen::TargetBackend::CpuSimd
    } else if cpp && !cuda {
        crate::nns::codegen::codegen::TargetBackend::CpuCxx
    } else if cuda {
        crate::nns::codegen::codegen::TargetBackend::Cuda
    } else if cpp {
        crate::nns::codegen::codegen::TargetBackend::CpuCxx
    } else {
        crate::nns::codegen::codegen::TargetBackend::Cuda
    };
    let opts = crate::nns::codegen::codegen::CodegenOptions {
        backend,
        emit_runtime_driver: runtime,
        enable_fp16: fp16,
        ..crate::nns::codegen::codegen::CodegenOptions::default()
    };

    let cg = crate::nns::codegen::codegen::CodeGenerator;
    match cg.generate(&module, &opts) {
        Ok(code) => {
            if let Some(out_path) = output {
                if let Err(e) = fs::write(&out_path, &code) {
                    eprintln!("Error writing {}: {}", out_path.display(), e);
                    process::exit(1);
                }
            } else {
                print!("{}", code);
            }
        }
        Err(e) => {
            let msg = e.0.clone();
            if msg.contains("cannot infer static shape")
                || msg.contains("dynamic dims are not supported")
                || msg.contains("Static shape")
            {
                // Surface codegen shape errors with the same header nsc uses
                let (line, col, m) = nns_parse_ns_error(&msg);
                // If the message already contains a location prefix, use parsed form;
                // otherwise fall back to raw codegen message under header.
                if m != msg || msg.contains(" at ") {
                    eprintln!("Static shape/type errors:");
                    eprintln!("  {}:{}  {}", line, col, m);
                } else {
                    eprintln!("Static shape/type errors:");
                    eprintln!("  1:1  {}", msg);
                }
            } else {
                eprintln!("Error: {}", e);
            }
            process::exit(1);
        }
    }
}
