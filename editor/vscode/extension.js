const vscode = require('vscode');
const { LanguageClient, TransportKind } = require('vscode-languageclient/node');

let client;

function activate(context) {
  const command = process.env.NAMEL3SS_LSP_COMMAND || 'namel3ss';
  const args = ['lsp'];

  const serverOptions = {
    run: {
      command,
      args,
      options: { env: process.env }
    },
    debug: {
      command,
      args,
      options: { env: process.env }
    },
    transport: TransportKind.stdio
  };

  const clientOptions = {
    documentSelector: [{ scheme: 'file', language: 'namel3ss' }],
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher('**/*.n3')
    }
  };

  client = new LanguageClient(
    'namel3ssLanguageServer',
    'Namel3ss Language Server',
    serverOptions,
    clientOptions
  );

  context.subscriptions.push(client.start());

  const restart = vscode.commands.registerCommand('namel3ss.restartLanguageServer', async () => {
    if (client) {
      await client.stop();
    }
    client.start();
    vscode.window.showInformationMessage('Namel3ss language server restarted');
  });
  context.subscriptions.push(restart);
}

function deactivate() {
  return client ? client.stop() : Promise.resolve();
}

module.exports = { activate, deactivate };
