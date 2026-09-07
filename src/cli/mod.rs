use clap::{Parser as ClapParser, Subcommand};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use crate::ast::*;
use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::typechecker::TypeChecker;
use crate::codegen::{compile_to_c, compile_to_c_tests};
use crate::modules::ModuleResolver;

#[derive(ClapParser)]
#[command(name = "glyphc")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "Glyph language compiler", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
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
}

pub fn run() {
    let cli = Cli::parse();

    match cli.command {
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
            run_tests(input.as_ref(), &compiler, &opt);
        }
    }
}

fn read_source(path: &PathBuf) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading file {}: {}", path.display(), e);
        process::exit(1);
    })
}

fn compile_file(input: &PathBuf, output: &PathBuf, emit_ir: bool, no_typecheck: bool) {
    if let Err(e) = compile_file_opts(input, output, emit_ir, no_typecheck, None, false) {
        eprintln!("{}", e);
        process::exit(1);
    }
}

fn compile_file_opts(
    input: &PathBuf,
    output: &PathBuf,
    emit_ir: bool,
    no_typecheck: bool,
    test_names: Option<&[String]>,
    quiet: bool,
) -> Result<(), String> {
    // Resolve modules
    let root_dir = input.parent().and_then(|p| p.parent()).unwrap_or(Path::new(".")).to_path_buf();
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
        compile_to_c_tests(&merged_program, output.to_str().unwrap(), tests)
    } else {
        compile_to_c(&merged_program, output.to_str().unwrap())
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
                type_params: type_params.clone(),
                params: params.clone(),
                return_type: return_type.clone(),
                body: body.clone(),
                is_async: *is_async,
                is_test: *is_test,
                pub_vis: *pub_vis,
            }
        }
        TopLevelItem::Struct { name, fields, pub_vis } => {
            TopLevelItem::Struct {
                name: format!("{}{}", prefix, name),
                fields: fields.clone(),
                pub_vis: *pub_vis,
            }
        }
        TopLevelItem::Enum { name, variants, pub_vis } => {
            TopLevelItem::Enum {
                name: format!("{}{}", prefix, name),
                variants: variants.clone(),
                pub_vis: *pub_vis,
            }
        }
        TopLevelItem::Const { name, ty, value, pub_vis } => {
            TopLevelItem::Const {
                name: format!("{}{}", prefix, name),
                ty: ty.clone(),
                value: value.clone(),
                pub_vis: *pub_vis,
            }
        }
        TopLevelItem::Impl { type_name, methods, pub_vis } => {
            TopLevelItem::Impl {
                type_name: format!("{}{}", prefix, type_name),
                methods: methods.clone(),
                pub_vis: *pub_vis,
            }
        }
        other => other.clone(),
    }
}

/// Rewrite unqualified references to the module's own top-level names
/// (functions/consts) inside its item bodies after prefixing.
fn rewrite_self_references(
    item: &mut TopLevelItem,
    self_names: &HashSet<String>,
    prefix: &str,
) {
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
        Stmt::Assignment { target, value } => {
            remap_expr(target, self_names, prefix);
            remap_expr(value, self_names, prefix);
        }
        Stmt::Expression(expr) => remap_expr(expr, self_names, prefix),
        Stmt::Return(Some(expr)) => remap_expr(expr, self_names, prefix),
        Stmt::Loop(body) => {
            for s in body {
                remap_stmt(s, self_names, prefix);
            }
        }
        Stmt::While { condition, body } => {
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
        Stmt::Guard { condition, else_body } => {
            remap_expr(condition, self_names, prefix);
            for s in else_body {
                remap_stmt(s, self_names, prefix);
            }
        }
        Stmt::Spawn(expr) => remap_expr(expr, self_names, prefix),
        _ => {}
    }
}

fn remap_expr(expr: &mut Expr, self_names: &HashSet<String>, prefix: &str) {
    match expr {
        Expr::IntegerLiteral(_)
        | Expr::FloatLiteral(_)
        | Expr::StringLiteral(_)
        | Expr::BoolLiteral(_) => {}
        Expr::Identifier(name) => {
            if self_names.contains(name) {
                *name = format!("{}{}", prefix, name);
            }
        }
        Expr::BinaryOp { left, right, .. } => {
            remap_expr(left, self_names, prefix);
            remap_expr(right, self_names, prefix);
        }
        Expr::UnaryOp { expr: inner, .. } => remap_expr(inner, self_names, prefix),
        Expr::Ref(inner) => remap_expr(inner, self_names, prefix),
        Expr::Cast { expr: inner, .. } => remap_expr(inner, self_names, prefix),
        Expr::FunctionCall { name, args } => {
            remap_expr(name, self_names, prefix);
            for arg in args {
                remap_expr(arg, self_names, prefix);
            }
        }
        Expr::MethodCall {
            object,
            args,
            ..
        } => {
            remap_expr(object, self_names, prefix);
            for arg in args {
                remap_expr(arg, self_names, prefix);
            }
        }
        Expr::FieldAccess { object, .. } => remap_expr(object, self_names, prefix),
        Expr::IndexAccess { object, index } => {
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
        Expr::Match { expr: inner, arms } => {
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
        } => {
            remap_expr(condition, self_names, prefix);
            remap_expr(then_branch, self_names, prefix);
            if let Some(else_branch) = else_branch {
                remap_expr(else_branch, self_names, prefix);
            }
        }
        Expr::Block(stmts) => {
            for stmt in stmts {
                remap_stmt(stmt, self_names, prefix);
            }
        }
        Expr::Range { start, end, .. } => {
            remap_expr(start, self_names, prefix);
            remap_expr(end, self_names, prefix);
        }
        Expr::ArrayLiteral(elements) => {
            for element in elements {
                remap_expr(element, self_names, prefix);
            }
        }
        Expr::ChannelBounded { capacity, .. } => remap_expr(capacity, self_names, prefix),
        Expr::Await(inner) => remap_expr(inner, self_names, prefix),
    }
}

fn check_file(input: &PathBuf) {
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
            eprintln!("Type error: {}", e);
            process::exit(1);
        }
    }
}

fn print_tokens(input: &PathBuf) {
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

fn print_ast(input: &PathBuf) {
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

fn run_file(input: &PathBuf, compiler: &str, opt: &str) {
    // Create temp directory
    let temp_dir = std::env::temp_dir().join("glyphc");
    fs::create_dir_all(&temp_dir).unwrap_or_else(|e| {
        eprintln!("Error creating temp dir: {}", e);
        process::exit(1);
    });

    let stem = input.file_stem().unwrap().to_str().unwrap();
    let c_file = temp_dir.join(format!("{}.c", stem));
    let binary = temp_dir.join(stem);

    // Compile Glyph to C
    compile_file(input, &c_file, false, false);

    // Compile C to binary
    let output = process::Command::new(compiler)
        .args([
            opt,
            "-std=gnu11",
            "-pthread",
            "-o",
            binary.to_str().unwrap(),
            c_file.to_str().unwrap(),
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
    let status = process::Command::new(binary.to_str().unwrap())
        .status()
        .unwrap_or_else(|e| {
            eprintln!("Error running binary: {}", e);
            process::exit(1);
        });

    if !status.success() {
        process::exit(status.code().unwrap_or(1));
    }

    // Cleanup
    let _ = fs::remove_file(&c_file);
    let _ = fs::remove_file(&binary);
}

fn run_tests(input: Option<&PathBuf>, compiler: &str, opt: &str) {
    let temp_dir = std::env::temp_dir().join("glyphc");
    fs::create_dir_all(&temp_dir).unwrap_or_else(|e| {
        eprintln!("Error creating temp dir: {}", e);
        process::exit(1);
    });

    // Collect candidate .glyph files
    let mut files: Vec<PathBuf> = Vec::new();
    if let Some(path) = input {
        files.push(path.clone());
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

    for file in &files {
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

        let stem = file.file_stem().unwrap().to_str().unwrap();
        let c_file = temp_dir.join(format!("{}.test.c", stem));
        let binary = temp_dir.join(format!("{}.test", stem));

        println!("{}Running {} test(s) in {}...{}", yellow, test_names.len(), file.display(), reset);

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
                binary.to_str().unwrap(),
                c_file.to_str().unwrap(),
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
        let run = process::Command::new(binary.to_str().unwrap())
            .output()
            .unwrap_or_else(|e| {
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
                    println!("  {}[FAILED]{} {}", red, reset, name.trim_end_matches(" FAILED"));
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
        if file_failed > 0 || run.status.code().map_or(true, |c| c != 0) {
            for line in &body_lines {
                println!("    {}", line);
            }
        }

        total_passed += file_passed;
        total_failed += file_failed;
        total_files += 1;
        if file_failed > 0 || run.status.code().map_or(true, |c| c != 0) {
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
            } else if path.extension().map_or(false, |e| e == "glyph") {
                out.push(path);
            }
        }
    }
}

/// Parse a file and return the names of its @test functions (in declaration order).
fn extract_test_functions(path: &PathBuf) -> Result<Vec<String>, String> {
    let source = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().map_err(|e| format!("Lexer error: {}", e))?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| format!("Parse error: {}", e))?;

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
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().unwrap_or_default() == "glyph" {
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
        let stem = glyph_file.file_stem().unwrap().to_str().unwrap();
        let c_file = build_dir.join(format!("{}.c", stem));

        println!("Compiling {} -> {}", glyph_file.display(), c_file.display());
        compile_file(glyph_file, &c_file, false, false);
        c_files.push(c_file);
    }

    // Link all C files
    let output_name = "main";
    let output_path = build_dir.join(output_name);

    let mut args = vec![optimization.as_str(), "-std=gnu11", "-pthread", "-o", output_path.to_str().unwrap()];
    for c_file in &c_files {
        args.push(c_file.to_str().unwrap());
    }
    args.push("-lm");

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
