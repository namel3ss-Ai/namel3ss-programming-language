/**
 * Testing Provider - AI-powered test generation
 */
import * as vscode from 'vscode';
import { AiProvider } from './aiProvider';
export declare class TestingProvider {
    private aiProvider;
    constructor(aiProvider: AiProvider);
    generateTests(document: vscode.TextDocument): Promise<void>;
    private createTestFile;
}
//# sourceMappingURL=testingProvider.d.ts.map