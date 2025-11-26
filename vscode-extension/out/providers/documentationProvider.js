"use strict";
/**
 * Documentation Provider - AI-powered documentation generation
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
exports.DocumentationProvider = void 0;
const vscode = __importStar(require("vscode"));
class DocumentationProvider {
    constructor(aiProvider) {
        this.aiProvider = aiProvider;
    }
    async generateDocumentation(document) {
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
        }
        catch (error) {
            vscode.window.showErrorMessage(`Failed to generate documentation: ${error}`);
        }
    }
    async createDocumentationFile(sourceDocument, docs) {
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
exports.DocumentationProvider = DocumentationProvider;
//# sourceMappingURL=documentationProvider.js.map