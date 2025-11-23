/**
 * Documentation Provider - AI-powered documentation generation
 */

import * as vscode from 'vscode';
import { AiProvider } from './aiProvider';

export class DocumentationProvider {
    constructor(private aiProvider: AiProvider) {}

    async generateDocumentation(document: vscode.TextDocument): Promise<void> {
        try {
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'Generating documentation...',
                cancellable: false
            }, async (progress) => {
                progress.report({ increment: 0, message: 'Analyzing code structure...' });

                const code = document.getText();
                
                progress.report({ increment: 40, message: 'Generating documentation...' });

                const docs = await this.aiProvider.generateDocumentation(code);
                
                progress.report({ increment: 80, message: 'Creating documentation file...' });

                await this.createDocumentationFile(document, docs);
                
                progress.report({ increment: 100, message: 'Documentation generated!' });
            });

        } catch (error) {
            vscode.window.showErrorMessage(`Failed to generate documentation: ${error}`);
        }
    }

    private async createDocumentationFile(sourceDocument: vscode.TextDocument, docs: string): Promise<void> {
        const sourceFileName = sourceDocument.fileName;
        const docFileName = sourceFileName.replace(/\.n3$/, '.md').replace(/.*\//, 'docs/');
        
        const docDoc = await vscode.workspace.openTextDocument({
            content: docs,
            language: 'markdown'
        });

        await vscode.window.showTextDocument(docDoc);
        
        // Suggest saving the file
        const saveUri = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.file(docFileName),
            filters: {
                'Markdown': ['md'],
                'All Files': ['*']
            }
        });

        if (saveUri) {
            await vscode.workspace.fs.writeFile(saveUri, Buffer.from(docs));
            await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
            await vscode.window.showTextDocument(saveUri);
            
            vscode.window.showInformationMessage('📚 Documentation created successfully!');
        }
    }
}