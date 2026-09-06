use crate::ast::*;
use crate::lexer::Lexer;
use crate::parser::{Parser, ParseError};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModuleError {
    #[error("Module not found: {0}")]
    ModuleNotFound(String),

    #[error("Cyclic import detected: {0}")]
    CyclicImport(String),

    #[error("Parse error in module {0}: {1}")]
    ParseError(String, ParseError),

    #[error("File I/O error: {0}")]
    IoError(String),

    #[error("Duplicate module: {0}")]
    DuplicateModule(String),
}

/// Represents a resolved module with its path and parsed AST
#[derive(Debug, Clone)]
pub struct ResolvedModule {
    pub name: String,
    pub path: PathBuf,
    pub ast: Program,
    pub dependencies: Vec<String>,
}

/// Dependency graph node for Tarjan's algorithm
#[derive(Debug, Clone)]
struct GraphNode {
    index: Option<usize>,
    lowlink: Option<usize>,
    on_stack: bool,
}

/// Module resolver and dependency graph
pub struct ModuleResolver {
    root_dir: PathBuf,
    src_dir: PathBuf,
    modules: HashMap<String, ResolvedModule>,
    graph: HashMap<String, Vec<String>>,
}

impl ModuleResolver {
    pub fn new(root_dir: PathBuf) -> Self {
        let src_dir = root_dir.join("src");
        ModuleResolver {
            root_dir,
            src_dir,
            modules: HashMap::new(),
            graph: HashMap::new(),
        }
    }

    /// Resolve all modules starting from the main file
    pub fn resolve(&mut self, main_path: &Path) -> Result<(), ModuleError> {
        let main_name = self.path_to_module_name(main_path);
        self.resolve_module_recursive(&main_name, main_path)?;
        Ok(())
    }

    /// Recursively resolve a module and its dependencies
    fn resolve_module_recursive(
        &mut self,
        module_name: &str,
        file_path: &Path,
    ) -> Result<(), ModuleError> {
        if self.modules.contains_key(module_name) {
            return Ok(());
        }

        // Read and parse the file
        let source = std::fs::read_to_string(file_path)
            .map_err(|e| ModuleError::IoError(format!("{}: {}", file_path.display(), e)))?;

        let mut lexer = Lexer::new(&source);
        let tokens = lexer
            .tokenize()
            .map_err(|e| ModuleError::ParseError(module_name.to_string(), ParseError::LexerError(e)))?;

        let mut parser = Parser::new(tokens);
        let ast = parser
            .parse_program()
            .map_err(|e| ModuleError::ParseError(module_name.to_string(), e))?;

        // Extract dependencies from @use statements
        let dependencies = self.extract_dependencies(&ast);

        // Store the module
        self.modules.insert(
            module_name.to_string(),
            ResolvedModule {
                name: module_name.to_string(),
                path: file_path.to_path_buf(),
                ast,
                dependencies: dependencies.clone(),
            },
        );

        // Build dependency graph
        self.graph
            .insert(module_name.to_string(), dependencies.clone());

        // Recursively resolve dependencies
        for dep_name in &dependencies {
            let dep_path = self.find_module_file(dep_name)?;
            self.resolve_module_recursive(dep_name, &dep_path)?;
        }

        Ok(())
    }

    /// Extract @use dependencies from AST
    fn extract_dependencies(&self, ast: &Program) -> Vec<String> {
        let mut deps = Vec::new();
        for item in &ast.items {
            if let TopLevelItem::Use { path } = item {
                let dep_name = path.join("::");
                deps.push(dep_name);
            }
        }
        deps
    }

    /// Find the file path for a module name
    fn find_module_file(&self, module_name: &str) -> Result<PathBuf, ModuleError> {
        // Convert module name to path: std::math -> src/std/math.glyph
        let relative_path = module_name.replace("::", "/");
        let file_path = self.src_dir.join(format!("{}.glyph", relative_path));

        if file_path.exists() {
            return Ok(file_path);
        }

        // Try as a file directly in src/
        let direct_path = self.src_dir.join(format!("{}.glyph", module_name));
        if direct_path.exists() {
            return Ok(direct_path);
        }

        Err(ModuleError::ModuleNotFound(module_name.to_string()))
    }

    /// Convert a file path to a module name
    pub fn path_to_module_name(&self, path: &Path) -> String {
        let relative = path.strip_prefix(&self.src_dir).unwrap_or(path);
        let without_ext = relative.with_extension("");
        without_ext
            .components()
            .map(|c| c.as_os_str().to_str().unwrap())
            .collect::<Vec<_>>()
            .join("::")
    }

    /// Detect cycles using Tarjan's algorithm
    pub fn detect_cycles(&self) -> Result<(), ModuleError> {
        let mut index_counter = 0;
        let mut stack: Vec<String> = Vec::new();
        let mut on_stack: HashSet<String> = HashSet::new();
        let mut indices: HashMap<String, usize> = HashMap::new();
        let mut lowlinks: HashMap<String, usize> = HashMap::new();

        for module_name in self.graph.keys() {
            if !indices.contains_key(module_name) {
                self.tarjan_dfs(
                    module_name,
                    &mut index_counter,
                    &mut stack,
                    &mut on_stack,
                    &mut indices,
                    &mut lowlinks,
                )?;
            }
        }

        Ok(())
    }

    /// Tarjan's DFS for cycle detection
    fn tarjan_dfs(
        &self,
        node: &str,
        index_counter: &mut usize,
        stack: &mut Vec<String>,
        on_stack: &mut HashSet<String>,
        indices: &mut HashMap<String, usize>,
        lowlinks: &mut HashMap<String, usize>,
    ) -> Result<(), ModuleError> {
        indices.insert(node.to_string(), *index_counter);
        lowlinks.insert(node.to_string(), *index_counter);
        *index_counter += 1;
        stack.push(node.to_string());
        on_stack.insert(node.to_string());

        // Visit neighbors
        if let Some(neighbors) = self.graph.get(node) {
            for neighbor in neighbors {
                if !indices.contains_key(neighbor) {
                    // Neighbor not yet visited
                    self.tarjan_dfs(
                        neighbor,
                        index_counter,
                        stack,
                        on_stack,
                        indices,
                        lowlinks,
                    )?;
                    let neighbor_lowlink = *lowlinks.get(neighbor).unwrap();
                    let node_lowlink = lowlinks.get(node).copied().unwrap_or(0);
                    lowlinks.insert(node.to_string(), node_lowlink.min(neighbor_lowlink));
                } else if on_stack.contains(neighbor) {
                    // Neighbor is on the stack - back edge (cycle)
                    let neighbor_index = *indices.get(neighbor).unwrap();
                    let node_lowlink = lowlinks.get(node).copied().unwrap_or(0);
                    lowlinks.insert(node.to_string(), node_lowlink.min(neighbor_index));
                }
            }
        }

        // If node is a root node, pop the stack to extract the SCC
        let node_index = *indices.get(node).unwrap();
        let node_lowlink = *lowlinks.get(node).unwrap();
        if node_index == node_lowlink {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                scc.push(w.clone());
                if w == node {
                    break;
                }
            }

            // If SCC has more than one node, it's a cycle
            if scc.len() > 1 {
                return Err(ModuleError::CyclicImport(scc.join(" -> ")));
            }
        }

        Ok(())
    }

    /// Topological sort using Kahn's algorithm
    pub fn topological_sort(&self) -> Result<Vec<String>, ModuleError> {
        // First check for cycles
        self.detect_cycles()?;

        // Build reverse graph: rev[b] = [a, ...] means a depends on b
        let mut rev: HashMap<String, Vec<String>> = HashMap::new();
        for (name, deps) in &self.graph {
            rev.entry(name.clone()).or_insert_with(Vec::new);
            for dep in deps {
                rev.entry(dep.clone()).or_insert_with(Vec::new).push(name.clone());
            }
        }

        // Calculate in-degrees using the original graph
        // in_degree[a] = number of things a depends on (outgoing edges)
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        for name in self.graph.keys() {
            let degree = self.graph.get(name).map_or(0, |d| d.len());
            in_degree.insert(name.clone(), degree);
        }

        // Start with nodes that have no dependencies
        let mut queue: VecDeque<String> = VecDeque::new();
        for (name, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(name.clone());
            }
        }

        let mut sorted = Vec::new();

        while let Some(node) = queue.pop_front() {
            sorted.push(node.clone());

            // For each module that depends on this node, reduce its in-degree
            if let Some(dependents) = rev.get(&node) {
                for dep in dependents {
                    let degree = in_degree.get_mut(dep).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(dep.clone());
                    }
                }
            }
        }

        if sorted.len() != self.graph.len() {
            return Err(ModuleError::CyclicImport(
                "Not all modules could be sorted".to_string(),
            ));
        }

        Ok(sorted)
    }

    /// Get all resolved modules
    pub fn get_modules(&self) -> &HashMap<String, ResolvedModule> {
        &self.modules
    }

    /// Get a specific module
    pub fn get_module(&self, name: &str) -> Option<&ResolvedModule> {
        self.modules.get(name)
    }

    /// Get the root directory
    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    /// Get the src directory
    pub fn src_dir(&self) -> &Path {
        &self.src_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_sort_simple() {
        let mut resolver = ModuleResolver::new(PathBuf::from("."));
        resolver
            .graph
            .insert("a".to_string(), vec!["b".to_string()]);
        resolver
            .graph
            .insert("b".to_string(), vec!["c".to_string()]);
        resolver
            .graph
            .insert("c".to_string(), vec![]);

        let sorted = resolver.topological_sort().unwrap();
        assert!(sorted.iter().position(|x| x == "c") < sorted.iter().position(|x| x == "b"));
        assert!(sorted.iter().position(|x| x == "b") < sorted.iter().position(|x| x == "a"));
    }

    #[test]
    fn test_cycle_detection() {
        let mut resolver = ModuleResolver::new(PathBuf::from("."));
        resolver
            .graph
            .insert("a".to_string(), vec!["b".to_string()]);
        resolver
            .graph
            .insert("b".to_string(), vec!["a".to_string()]);

        let result = resolver.detect_cycles();
        assert!(result.is_err());
    }

    #[test]
    fn test_no_cycle() {
        let mut resolver = ModuleResolver::new(PathBuf::from("."));
        resolver
            .graph
            .insert("a".to_string(), vec!["b".to_string()]);
        resolver
            .graph
            .insert("b".to_string(), vec!["c".to_string()]);
        resolver
            .graph
            .insert("c".to_string(), vec![]);

        let result = resolver.detect_cycles();
        assert!(result.is_ok());
    }
}
