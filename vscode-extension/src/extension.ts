/**
 * Namel3ss AI Assistant VS Code Extension
 * Main extension entry point with AI-powered development features
 */

import * as vscode from 'vscode';
import { AiProvider } from './providers/aiProvider';
import { ChatProvider } from './providers/chatProvider';
import { CodeGenerationProvider } from './providers/codeGenerationProvider';
import { CompletionProvider } from './providers/completionProvider';
import { RefactoringProvider } from './providers/refactoringProvider';
import { TestingProvider } from './providers/testingProvider';
import { DocumentationProvider } from './providers/documentationProvider';
import { LanguageServerClient } from './languageServer/client';

let aiProvider: AiProvider;
let chatProvider: ChatProvider;
let codeGenerationProvider: CodeGenerationProvider;
let languageClient: LanguageServerClient;

export function activate(context: vscode.ExtensionContext) {
    console.log('🚀 Namel3ss AI Assistant is activating...');

    // Initialize AI provider
    aiProvider = new AiProvider(context);
    
    // Initialize language server client
    languageClient = new LanguageServerClient();
    languageClient.start();

    // Initialize providers
    chatProvider = new ChatProvider(context, aiProvider);
    codeGenerationProvider = new CodeGenerationProvider(context, aiProvider);
    const completionProvider = new CompletionProvider(aiProvider);
    const refactoringProvider = new RefactoringProvider(aiProvider);
    const testingProvider = new TestingProvider(aiProvider);
    const documentationProvider = new DocumentationProvider(aiProvider);

    // Register commands
    registerCommands(context, {
        chatProvider,
        codeGenerationProvider,
        completionProvider,
        refactoringProvider,
        testingProvider,
        documentationProvider
    });

    // Register providers
    registerProviders(context, {
        completionProvider,
        refactoringProvider
    });

    // Register views
    registerViews(context, {
        chatProvider,
        codeGenerationProvider,
        refactoringProvider
    });

    // Show activation message
    vscode.window.showInformationMessage('🎉 Namel3ss AI Assistant activated!');
    
    console.log('✅ Namel3ss AI Assistant activated successfully');
}

export function deactivate(): Thenable<void> | undefined {
    console.log('🛑 Deactivating Namel3ss AI Assistant...');
    
    if (languageClient) {
        return languageClient.stop();
    }
    
    return undefined;
}

function registerCommands(context: vscode.ExtensionContext, providers: any) {
    const commands = [
        // Chat commands
        vscode.commands.registerCommand('namel3ss.openChat', () => {
            providers.chatProvider.openChat();
        }),

        // Code generation commands  
        vscode.commands.registerCommand('namel3ss.generateCode', async () => {
            const description = await vscode.window.showInputBox({
                prompt: 'Describe the component you want to generate',
                placeholder: 'e.g., "Create a user dashboard with charts and tables"'
            });
            
            if (description) {
                await providers.codeGenerationProvider.generateComponent(description);
            }
        }),

        vscode.commands.registerCommand('namel3ss.generatePage', async () => {
            const description = await vscode.window.showInputBox({
                prompt: 'Describe the page you want to generate',
                placeholder: 'e.g., "Create a settings page with form controls"'
            });
            
            if (description) {
                await providers.codeGenerationProvider.generatePage(description);
            }
        }),

        // Code explanation
        vscode.commands.registerCommand('namel3ss.explainCode', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.selection;
            const text = editor.document.getText(selection.isEmpty ? undefined : selection);
            
            if (text.trim()) {
                const explanation = await providers.codeGenerationProvider.explainCode(text);
                vscode.window.showInformationMessage(explanation, { modal: true });
            }
        }),

        // Refactoring commands
        vscode.commands.registerCommand('namel3ss.refactorCode', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const suggestions = await providers.refactoringProvider.getRefactoringSuggestions(
                editor.document.getText(),
                editor.selection
            );

            if (suggestions.length > 0) {
                const choice = await vscode.window.showQuickPick(
                    suggestions.map(s => ({ label: s.title, detail: s.description, suggestion: s })),
                    { placeHolder: 'Select a refactoring suggestion' }
                );

                if (choice) {
                    await providers.refactoringProvider.applyRefactoring(choice.suggestion, editor);
                }
            } else {
                vscode.window.showInformationMessage('No refactoring suggestions available for this code.');
            }
        }),

        // Testing commands
        vscode.commands.registerCommand('namel3ss.generateTests', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'namel3ss') {
                vscode.window.showErrorMessage('Please open a Namel3ss file to generate tests.');
                return;
            }

            await providers.testingProvider.generateTests(editor.document);
        }),

        // Documentation commands
        vscode.commands.registerCommand('namel3ss.generateDocs', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            await providers.documentationProvider.generateDocumentation(editor.document);
        }),

        // Performance optimization
        vscode.commands.registerCommand('namel3ss.optimizePerformance', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const suggestions = await providers.refactoringProvider.getPerformanceOptimizations(
                editor.document.getText()
            );

            if (suggestions.length > 0) {
                const panel = vscode.window.createWebviewPanel(
                    'namel3ssOptimizations',
                    'Performance Optimizations',
                    vscode.ViewColumn.Beside,
                    { enableScripts: true }
                );

                panel.webview.html = providers.refactoringProvider.getOptimizationWebview(suggestions);
            }
        }),

        // Quick actions
        vscode.commands.registerCommand('namel3ss.quickFix', async (uri: vscode.Uri, range: vscode.Range) => {
            const document = await vscode.workspace.openTextDocument(uri);
            const text = document.getText(range);
            
            const fixes = await providers.refactoringProvider.getQuickFixes(text, document, range);
            
            if (fixes.length > 0) {
                const choice = await vscode.window.showQuickPick(
                    fixes.map(f => ({ label: f.title, detail: f.description, fix: f })),
                    { placeHolder: 'Select a quick fix' }
                );

                if (choice) {
                    const editor = await vscode.window.showTextDocument(document);
                    await providers.refactoringProvider.applyQuickFix(choice.fix, editor);
                }
            }
        })
    ];

    commands.forEach(command => context.subscriptions.push(command));
}

function registerProviders(context: vscode.ExtensionContext, providers: any) {
    // Register completion provider
    const completionDisposable = vscode.languages.registerCompletionItemProvider(
        { language: 'namel3ss' },
        providers.completionProvider,
        '.', ' ', '(', '"', "'"
    );

    // Register hover provider
    const hoverDisposable = vscode.languages.registerHoverProvider(
        { language: 'namel3ss' },
        providers.refactoringProvider
    );

    // Register code action provider
    const codeActionDisposable = vscode.languages.registerCodeActionsProvider(
        { language: 'namel3ss' },
        providers.refactoringProvider,
        {
            providedCodeActionKinds: [
                vscode.CodeActionKind.QuickFix,
                vscode.CodeActionKind.Refactor,
                vscode.CodeActionKind.RefactorExtract,
                vscode.CodeActionKind.RefactorInline,
                vscode.CodeActionKind.RefactorRewrite
            ]
        }
    );

    context.subscriptions.push(completionDisposable, hoverDisposable, codeActionDisposable);
}

function registerViews(context: vscode.ExtensionContext, providers: any) {
    // Register chat webview provider
    const chatViewProvider = vscode.window.registerWebviewViewProvider(
        'namel3ssChat',
        providers.chatProvider
    );

    // Register code generation tree provider
    const codeGenTreeProvider = vscode.window.registerTreeDataProvider(
        'namel3ssCodeGen',
        providers.codeGenerationProvider
    );

    // Register refactoring tree provider
    const refactorTreeProvider = vscode.window.registerTreeDataProvider(
        'namel3ssRefactor',
        providers.refactoringProvider.getTreeProvider()
    );

    context.subscriptions.push(chatViewProvider, codeGenTreeProvider, refactorTreeProvider);
}

// Context checking
vscode.workspace.onDidChangeWorkspaceFolders(() => {
    checkNamel3ssWorkspace();
});

vscode.workspace.onDidOpenTextDocument(() => {
    checkNamel3ssWorkspace();
});

function checkNamel3ssWorkspace() {
    const hasNamel3ssFiles = vscode.workspace.textDocuments.some(doc => doc.languageId === 'namel3ss') ||
                             vscode.workspace.findFiles('**/*.n3', null, 1);
    
    vscode.commands.executeCommand('setContext', 'workspaceHasNamel3ssFiles', hasNamel3ssFiles);
}