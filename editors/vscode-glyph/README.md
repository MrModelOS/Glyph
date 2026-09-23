# Glyph for Visual Studio Code

This extension provides syntax highlighting for `.glyph` and `.ns` files and
connects `.glyph` documents to the `glyphc` language server. NeuralScript files
receive highlighting; use `glyphc nns --check` for tensor diagnostics until a
dedicated NNS LSP mode is added.

## Development

Install dependencies and package the extension with:

```bash
cd editors/vscode-glyph
npm install
npm run package
```

The extension expects `glyphc` on `PATH`. To use a checkout, set
`glyphc.path` in VS Code settings or ensure `target/debug/glyphc` is on `PATH`.
The server is started as `glyphc lsp`; diagnostics, hover, completion, and
go-to-definition are provided by the compiler's built-in LSP.
