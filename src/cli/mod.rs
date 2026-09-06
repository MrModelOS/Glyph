use clap::{Parser as ClapParser, Subcommand};
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use crate::ast::*;
use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::typechecker::TypeChecker;
use crate::codegen::compile_to_c;
use crate::modules::{ModuleResolver, ModuleError};

#[derive(ClapParser)]
#[command(name = "glyphc")]
#[command(version = "0.1.0")]
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
        #[arg(short, long, default_value = "output.ll")]
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
    }
}

fn read_source(path: &PathBuf) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading file {}: {}", path.display(), e);
        process::exit(1);
    })
}

fn compile_file(input: &PathBuf, output: &PathBuf, emit_ir: bool, no_typecheck: bool) {
    // Resolve modules
    let root_dir = input.parent().and_then(|p| p.parent()).unwrap_or(Path::new(".")).to_path_buf();
    let mut resolver = ModuleResolver::new(root_dir.clone());

    if let Err(e) = resolver.resolve(input) {
        eprintln!("Module error: {}", e);
        process::exit(1);
    }

    // Get topological order
    let sorted = match resolver.topological_sort() {
        Ok(order) => order,
        Err(e) => {
            eprintln!("Module error: {}", e);
            process::exit(1);
        }
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

            for item in &module.ast.items {
                let prefixed = prefix_item(item, &prefix);
                all_items.push(prefixed);
            }
        }
    }

    let merged_program = Program { items: all_items };

    // Type checking
    if !no_typecheck {
        let mut checker = TypeChecker::new();
        if let Err(e) = checker.check_program(&merged_program) {
            eprintln!("Type error: {}", e);
            process::exit(1);
        }
    }

    // Code generation
    if emit_ir {
        println!("// C code for {}", input.display());
        println!("// ---");
    }

    match compile_to_c(&merged_program, output.to_str().unwrap()) {
        Ok(()) => {
            if emit_ir {
                println!("// C code written to {}", output.display());
            } else {
                println!("Compiled successfully to {}", output.display());
            }
        }
        Err(e) => {
            eprintln!("Code generation error: {}", e);
            process::exit(1);
        }
    }
}

fn prefix_item(item: &TopLevelItem, prefix: &str) -> TopLevelItem {
    match item {
        TopLevelItem::Function { name, params, return_type, body, is_async, pub_vis } => {
            // Don't prefix main function - it's the entry point
            let new_name = if name == "main" {
                name.clone()
            } else {
                format!("{}{}", prefix, name)
            };
            TopLevelItem::Function {
                name: new_name,
                params: params.clone(),
                return_type: return_type.clone(),
                body: body.clone(),
                is_async: *is_async,
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

    // Run the binary
    let output = process::Command::new(binary.to_str().unwrap())
        .output()
        .unwrap_or_else(|e| {
            eprintln!("Error running binary: {}", e);
            process::exit(1);
        });

    print!("{}", String::from_utf8_lossy(&output.stdout));

    if !output.status.success() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        process::exit(1);
    }

    // Cleanup
    let _ = fs::remove_file(&c_file);
    let _ = fs::remove_file(&binary);
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

    let mut args = vec![optimization.as_str(), "-std=gnu11", "-o", output_path.to_str().unwrap()];
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
