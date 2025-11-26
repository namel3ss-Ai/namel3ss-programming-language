/**
 * Completion Provider - AI-powered intelligent code completion
 */
import * as vscode from 'vscode';
import { AiProvider } from './aiProvider';
export declare class CompletionProvider implements vscode.CompletionItemProvider {
    private aiProvider;
    constructor(aiProvider: AiProvider);
    provideCompletionItems(document: vscode.TextDocument, position: vscode.Position, token: vscode.CancellationToken, context: vscode.CompletionContext): Promise<vscode.CompletionItem[]>;
    private getBasicCompletions;
    private getAICompletions;
    private shouldUseAICompletion;
    private createCompletionItem;
    private getCompletionKind;
}
//# sourceMappingURL=completionProvider.d.ts.map