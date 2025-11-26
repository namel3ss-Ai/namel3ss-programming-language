/**
 * Refactoring Provider - AI-powered code refactoring and optimization
 */
import * as vscode from 'vscode';
import { AiProvider } from './aiProvider';
export interface RefactoringSuggestion {
    title: string;
    description: string;
    type: 'extract' | 'inline' | 'rename' | 'optimize' | 'fix';
    priority: 'low' | 'medium' | 'high';
    range: vscode.Range;
    newCode: string;
}
export declare class RefactoringProvider implements vscode.CodeActionProvider, vscode.HoverProvider {
    private aiProvider;
    constructor(aiProvider: AiProvider);
    provideCodeActions(document: vscode.TextDocument, range: vscode.Range | vscode.Selection, context: vscode.CodeActionContext, token: vscode.CancellationToken): Promise<vscode.CodeAction[]>;
    provideHover(document: vscode.TextDocument, position: vscode.Position, token: vscode.CancellationToken): Promise<vscode.Hover | null>;
    getRefactoringSuggestions(code: string, selection?: vscode.Range): Promise<RefactoringSuggestion[]>;
    getQuickFixes(code: string, document: vscode.TextDocument, range: vscode.Range): Promise<RefactoringSuggestion[]>;
    getPerformanceOptimizations(code: string): Promise<RefactoringSuggestion[]>;
    applyRefactoring(suggestion: RefactoringSuggestion, editor: vscode.TextEditor): Promise<void>;
    applyQuickFix(fix: RefactoringSuggestion, editor: vscode.TextEditor): Promise<void>;
    getOptimizationWebview(suggestions: RefactoringSuggestion[]): string;
    getTreeProvider(): {
        getChildren: () => Promise<{
            label: string;
            description: string;
        }[]>;
        getTreeItem: (element: any) => any;
    };
    private getHoverText;
    private getCodeActionKind;
    private createWorkspaceEdit;
}
//# sourceMappingURL=refactoringProvider.d.ts.map