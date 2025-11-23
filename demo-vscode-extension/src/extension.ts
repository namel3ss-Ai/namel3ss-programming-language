import * as vscode from 'vscode';
import { LanguageClient, LanguageClientOptions, ServerOptions } from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: vscode.ExtensionContext) {
    // Language server setup
    const serverOptions: ServerOptions = {
        command: 'python',
        args: ['-m', 'namel3ss.lsp.server'],
        options: {
            env: { ...process.env }
        }
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'n3' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.n3')
        }
    };

    client = new LanguageClient('n3', 'N3 Language Server', serverOptions, clientOptions);

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('n3.refactor.modernizeLegacy', modernizeLegacySyntax),
        vscode.commands.registerCommand('n3.refactor.organizeStructure', organizeFileStructure),
        vscode.commands.registerCommand('n3.refactor.extractComponent', extractComponent),
        vscode.commands.registerCommand('n3.format.sortProperties', sortProperties),
    );

    // Start language client
    client.start();
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}

async function modernizeLegacySyntax() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'n3') {
        return;
    }

    const result = await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: "Modernizing N3 syntax...",
        cancellable: false
    }, async (progress) => {
        progress.report({ increment: 0 });
        
        // Execute refactoring command through LSP
        const workspaceEdit = await vscode.commands.executeCommand(
            'namel3ss.refactor.modernizeLegacy',
            editor.document.uri.toString()
        ) as vscode.WorkspaceEdit;
        
        progress.report({ increment: 100 });
        return workspaceEdit;
    });

    if (result && result.size > 0) {
        await vscode.workspace.applyEdit(result);
        vscode.window.showInformationMessage('Legacy syntax modernized successfully!');
    } else {
        vscode.window.showInformationMessage('No legacy syntax patterns found.');
    }
}

async function organizeFileStructure() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'n3') {
        return;
    }

    const workspaceEdit = await vscode.commands.executeCommand(
        'namel3ss.refactor.organizeStructure',
        editor.document.uri.toString()
    ) as vscode.WorkspaceEdit;

    if (workspaceEdit && workspaceEdit.size > 0) {
        await vscode.workspace.applyEdit(workspaceEdit);
        vscode.window.showInformationMessage('File structure organized!');
    }
}

async function extractComponent() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'n3' || editor.selection.isEmpty) {
        return;
    }

    const componentName = await vscode.window.showInputBox({
        prompt: 'Enter component name',
        placeHolder: 'my_component'
    });

    if (!componentName) {
        return;
    }

    const workspaceEdit = await vscode.commands.executeCommand(
        'namel3ss.refactor.extractComponent',
        editor.document.uri.toString(),
        editor.selection,
        componentName
    ) as vscode.WorkspaceEdit;

    if (workspaceEdit && workspaceEdit.size > 0) {
        await vscode.workspace.applyEdit(workspaceEdit);
        vscode.window.showInformationMessage(`Component "${componentName}" extracted successfully!`);
    }
}

async function sortProperties() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'n3') {
        return;
    }

    const workspaceEdit = await vscode.commands.executeCommand(
        'namel3ss.format.sortProperties',
        editor.document.uri.toString()
    ) as vscode.WorkspaceEdit;

    if (workspaceEdit && workspaceEdit.size > 0) {
        await vscode.workspace.applyEdit(workspaceEdit);
        vscode.window.showInformationMessage('Properties sorted alphabetically!');
    }
}