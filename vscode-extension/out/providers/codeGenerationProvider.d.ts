/**
 * Code Generation Provider - AI-powered code generation
 */
import * as vscode from 'vscode';
import { AiProvider } from './aiProvider';
export interface CodeGenItem {
    label: string;
    description: string;
    type: 'component' | 'page' | 'function' | 'app';
    command?: string;
}
export declare class CodeGenerationProvider implements vscode.TreeDataProvider<CodeGenItem> {
    private context;
    private aiProvider;
    private _onDidChangeTreeData;
    readonly onDidChangeTreeData: vscode.Event<CodeGenItem | undefined | null | void>;
    private items;
    constructor(context: vscode.ExtensionContext, aiProvider: AiProvider);
    getTreeItem(element: CodeGenItem): vscode.TreeItem;
    getChildren(element?: CodeGenItem): Thenable<CodeGenItem[]>;
    generateComponent(description: string): Promise<void>;
    generatePage(description: string): Promise<void>;
    explainCode(code: string): Promise<string>;
    private getWorkspaceContext;
    private createNewFile;
    private generateFilename;
    generateQuickComponent(type: 'dashboard' | 'form' | 'chart' | 'table'): Promise<void>;
    generateFromTemplate(templateName: string): Promise<void>;
    refresh(): void;
}
//# sourceMappingURL=codeGenerationProvider.d.ts.map