mod lexer;
mod ast;
mod parser;
mod typechecker;
mod codegen;
mod generics;
mod modules;
mod cli;
mod lsp;

use cli::run;

fn main() {
    // Check if running as LSP server
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--lsp" {
        let mut server = lsp::LspServer::new();
        server.run();
    } else {
        run();
    }
}
