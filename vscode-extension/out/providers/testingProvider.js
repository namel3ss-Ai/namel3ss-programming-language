"use strict";
/**
 * Testing Provider - AI-powered test generation
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.TestingProvider = void 0;
const vscode = __importStar(require("vscode"));
class TestingProvider {
    constructor(aiProvider) {
        this.aiProvider = aiProvider;
    }
    async generateTests(document) {
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
        }
        catch (error) {
            vscode.window.showErrorMessage(`Failed to generate tests: ${error}`);
        }
    }
    async createTestFile(sourceDocument, tests) {
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
exports.TestingProvider = TestingProvider;
//# sourceMappingURL=testingProvider.js.map