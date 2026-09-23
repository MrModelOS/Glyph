const vscode = require("vscode");
const { LanguageClient, TransportKind } = require("vscode-languageclient/node");

let client;

function startClient(context) {
  const config = vscode.workspace.getConfiguration("glyphc");
  const server = config.get("path", "glyphc");
  const serverOptions = {
    run: { command: server, args: ["lsp"] },
    debug: { command: server, args: ["lsp"] }
  };
  const clientOptions = {
    documentSelector: [{ scheme: "file", language: "glyph" }],
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher("**/*.glyph")
    }
  };

  client = new LanguageClient("glyphc", "Glyph Language Server", serverOptions, clientOptions);
  context.subscriptions.push(client.start());
}

function activate(context) {
  startClient(context);
  context.subscriptions.push(
    vscode.commands.registerCommand("glyph.restartServer", async () => {
      if (client) {
        await client.stop();
      }
      startClient(context);
    })
  );
}

function deactivate() {
  return client ? client.stop() : undefined;
}

module.exports = { activate, deactivate };
