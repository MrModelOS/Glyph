# Editor Setup

## VS Code

There is no marketplace release yet, but the repository contains a minimal
VS Code extension with the grammar and LSP client in
[`editors/vscode-glyph`](../editors/vscode-glyph/). You can also use the
standalone settings below.

### 1. File association (`settings.json`)

```json
{
  "files.associations": {
    "*.glyph": "glyph",
    "*.ns": "neural-script"
  },
  "[glyph]": {
    "editor.tabSize": 4,
    "editor.insertSpaces": true,
    "editor.wordWrap": "off"
  }
}
```

### 2. TextMate grammar stub

Create `.vscode/glyph.tmLanguage.json` in your workspace (or inside a local
extension `syntaxes/glyph.tmLanguage.json`):

```json
{
  "scopeName": "source.glyph",
  "name": "Glyph",
  "fileTypes": ["glyph"],
  "patterns": [
    { "include": "#comments" },
    { "include": "#attributes" },
    { "include": "#keywords" },
    { "include": "#types" },
    { "include": "#strings" },
    { "include": "#numbers" }
  ],
  "repository": {
    "comments": {
      "patterns": [
        { "name": "comment.line.double-slash.glyph", "match": "//.*$" },
        { "name": "comment.block.glyph", "begin": "/\\*", "end": "\\*/" }
      ]
    },
    "attributes": {
      "patterns": [
        {
          "name": "keyword.other.attribute.glyph",
          "match": "@(module|use|fn|struct|enum|impl|const|pub|test)\\b"
        },
        { "name": "keyword.other.guard.glyph", "match": "#guard\\b" }
      ]
    },
    "keywords": {
      "patterns": [
        {
          "name": "keyword.control.glyph",
          "match": "\\b(let|mut|if|else|match|while|loop|for|in|return|break|continue|spawn|await|select|timeout|default|as)\\b"
        },
        {
          "name": "constant.language.glyph",
          "match": "\\b(true|false|None|Some|Ok|Err)\\b"
        }
      ]
    },
    "types": {
      "patterns": [
        {
          "name": "entity.name.type.glyph",
          "match": "\\b(Int64|UInt64|Float64|Bool|String|Bytes|List|Map|Option|Result|Channel|Async|Void)\\b"
        }
      ]
    },
    "strings": {
      "patterns": [
        { "name": "string.quoted.double.glyph", "begin": "\"", "end": "\"", "patterns": [{ "include": "#escapes" }] }
      ]
    },
    "numbers": {
      "patterns": [
        { "name": "constant.numeric.glyph", "match": "\\b0x[0-9a-fA-F_]+|\\b\\d[\\d_]*\\.?\\d*[\\d_]*\\b" }
      ]
    },
    "escapes": {
      "patterns": [{ "name": "constant.character.escape.glyph", "match": "\\\\." }]
    }
  }
}
```

Add a matching `source.neural-script` grammar for `*.ns` if desired
(keywords: `network`, `layer`, `forward`, `train`, `grad`, `Tensor`, `Dynamic`,
`type`, plus dtype tokens).

### 3. LSP — `glyphc lsp` (or `glyphc --lsp`)

The compiler exposes a stdio LSP server. Both forms are supported:

```bash
glyphc lsp
# backward-compatible alias:
glyphc --lsp
```

Capabilities: `textDocumentSync: Full`, `completionProvider` (triggers `@ # . :`),
`hoverProvider`, `definitionProvider`, `diagnosticProvider`. Diagnostics cover
lexer / parser / typechecker with full spans (`line:col-col` on type errors,
`line:col` on lex/parse).

#### VS Code — `vscode-languageclient` config

Example `extension.js` for a local extension (or use
`vscode-languageclient` + `LanguageClient`):

```js
const { LanguageClient, TransportKind } = require('vscode-languageclient/node');
function activate(ctx) {
  const serverOptions = {
    command: 'glyphc',
    args: ['lsp'],
    transport: TransportKind.stdio
  };
  const clientOptions = {
    documentSelector: [{ scheme: 'file', language: 'glyph' }],
    synchronize: { fileEvents: [] }
  };
  const client = new LanguageClient('glyph', 'Glyph LSP', serverOptions, clientOptions);
  ctx.subscriptions.push(client.start());
}
exports.activate = activate;
```

Alternatively, wire `glyphc lsp` as a generic LSP via extensions like
`vscode-lsp-generic` or `lsp-bridge`:

```json
{
  "lsp.servers": {
    "glyph": {
      "command": "glyphc",
      "args": ["lsp"],
      "filetypes": ["glyph"]
    }
  }
}
```

Check it works: open a `.glyph` file with a type error — diagnostics should
appear on save/change (full-sync).

---

## Neovim (nvim-lspconfig)

`glyphc lsp` is a plain stdio server. Register it as a custom lspconfig
server. Requires `neovim/nvim-lspconfig`.

### Minimal `init.lua`

```lua
local lspconfig = require('lspconfig')
local configs = require('lspconfig.configs')

if not configs.glyph then
  configs.glyph = {
    default_config = {
      cmd = { 'glyphc', 'lsp' },
      filetypes = { 'glyph' },
      root_dir = function(fname)
        return lspconfig.util.root_pattern('glyph.toml', '.git')(fname)
          or lspconfig.util.path.dirname(fname)
      end,
      single_file_support = true,
      settings = {},
    },
  }
end

lspconfig.glyph.setup {
  on_attach = function(client, bufnr)
    -- optional: hover / goto-def keymaps
    vim.keymap.set('n', 'K', vim.lsp.buf.hover, { buffer = bufnr })
    vim.keymap.set('n', 'gd', vim.lsp.buf.definition, { buffer = bufnr })
    vim.keymap.set('n', '<C-Space>', function() vim.lsp.buf.completion() end,
                   { buffer = bufnr, mode = 'i' })
  end,
}

-- .glyph filetype
vim.filetype.add({ extension = { glyph = 'glyph', ns = 'neural_script' } })
```

With `lazy.nvim`:

```lua
{
  'neovim/nvim-lspconfig',
  config = function()
    -- same configs.glyph block as above
  end,
}
```

### Backward-compatible flag

Older configurations may still use `glyphc --lsp`; it is an alias for
`glyphc lsp`. New configurations should prefer the subcommand form.

### What you get

- Live diagnostics: lexer errors (unterminated string, bad number), parser
  errors (`Unexpected token`), type errors with full spans
  (`line:col-col: Undefined variable: nope`). See `src/lsp/mod.rs:analyze`.
- Hover (`textDocument/hover`): keyword docs for `select`, `spawn`, `await`,
  `Channel`, `Async`, `Result`, `Option`, etc.
- Completion (`textDocument/completion`): `@module`, `@fn`, `@struct`, `@enum`,
  `let`, `match`, `spawn`/`await`/`select`/`timeout`/`default`, …
- Go-to-definition (`textDocument/definition`): local `let` / param → definition
  in the same item, else global `fn`/`struct` name.

Limitations: diagnostics are per-file (no cross-module workspace analysis yet);
text sync is Full (whole buffer on each change).

---

## Troubleshooting

- `glyphc: command not found` — build first: `cargo build --release` → `target/release/glyphc` on `PATH`.
- No diagnostics: ensure `filetypes` includes `glyph` and the server is attached (`:LspInfo` in Neovim, Output panel in VS Code).
- Hover empty: cursor must be on a word (`word_at` uses `is_alphanumeric || _ || # || @`).
