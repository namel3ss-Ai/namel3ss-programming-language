/**
 * Documentation Provider - AI-powered documentation generation
 */
import * as vscode from 'vscode';
import { AiProvider } from './aiProvider';
export declare class DocumentationProvider {
    private aiProvider;
    constructor(aiProvider: AiProvider);
    generateDocumentation(document: vscode.TextDocument): Promise<void>;
    private createDocumentationFile;
}
//# sourceMappingURL=documentationProvider.d.ts.map