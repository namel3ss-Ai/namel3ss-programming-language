/**
 * Testing Provider - AI-powered test generation
 */

import * as vscode from 'vscode';
import { AiProvider } from './aiProvider';

export class TestingProvider {
    constructor(private aiProvider: AiProvider) {}

    async generateTests(document: vscode.TextDocument): Promise<void> {
        try {
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'Generating tests...',
                cancellable: false
            }, async (progress) => {
                progress.report({ increment: 0, message: 'Analyzing code...' });

                const code = document.getText();
                
                progress.report({ increment: 30, message: 'Generating test cases...' });

                const tests = await this.aiProvider.generateTests(code);
                
                progress.report({ increment: 80, message: 'Creating test file...' });

                await this.createTestFile(document, tests);
                
                progress.report({ increment: 100, message: 'Tests generated!' });
            });

        } catch (error) {
            vscode.window.showErrorMessage(`Failed to generate tests: ${error}`);
        }
    }

    private async createTestFile(sourceDocument: vscode.TextDocument, tests: string): Promise<void> {
        const sourceFileName = sourceDocument.fileName;
        const testFileName = sourceFileName.replace(/\.n3$/, '.test.n3');
        
        const testDoc = await vscode.workspace.openTextDocument({
            content: tests,
            language: 'namel3ss'
        });

        await vscode.window.showTextDocument(testDoc);
        
        // Suggest saving the file
        const saveUri = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.file(testFileName),
            filters: {
                'Namel3ss Tests': ['test.n3'],
                'All Files': ['*']
            }
        });

        if (saveUri) {
            await vscode.workspace.fs.writeFile(saveUri, Buffer.from(tests));
            await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
            await vscode.window.showTextDocument(saveUri);
            
            vscode.window.showInformationMessage('✅ Test file created successfully!');
        }
    }
}