/**
 * Chat Provider - AI Chat Interface for VS Code
 */
import * as vscode from 'vscode';
import { AiProvider } from './aiProvider';
export declare class ChatProvider implements vscode.WebviewViewProvider {
    private readonly context;
    private readonly aiProvider;
    static readonly viewType = "namel3ssChat";
    private _view?;
    private _messages;
    constructor(context: vscode.ExtensionContext, aiProvider: AiProvider);
    resolveWebviewView(webviewView: vscode.WebviewView, context: vscode.WebviewViewResolveContext, _token: vscode.CancellationToken): void;
    openChat(): Promise<void>;
    private handleUserMessage;
    private getCurrentContext;
    private addMessageToChat;
    private updateMessage;
    private showTypingIndicator;
    private clearChat;
    private exportChat;
    private _getHtmlForWebview;
}
//# sourceMappingURL=chatProvider.d.ts.map