"use strict";
/**
 * Code Generation Provider - AI-powered code generation
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
exports.CodeGenerationProvider = void 0;
const vscode = __importStar(require("vscode"));
class CodeGenerationProvider {
    constructor(context, aiProvider) {
        this.context = context;
        this.aiProvider = aiProvider;
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
        this.items = [
            {
                label: 'Generate Component',
                description: 'Create a new N3 component',
                type: 'component',
                command: 'namel3ss.generateCode'
            },
            {
                label: 'Generate Page',
                description: 'Create a new page with routing',
                type: 'page',
                command: 'namel3ss.generatePage'
            },
            {
                label: 'Generate Function',
                description: 'Create utility or data function',
                type: 'function'
            },
            {
                label: 'Generate Full App',
                description: 'Generate complete application',
                type: 'app'
            }
        ];
    }
    getTreeItem(element) {
        const item = new vscode.TreeItem(element.label, vscode.TreeItemCollapsibleState.None);
        item.description = element.description;
        item.tooltip = element.description;
        // Set icons based on type
        switch (element.type) {
            case 'component':
                item.iconPath = new vscode.ThemeIcon('symbol-class');
                break;
            case 'page':
                item.iconPath = new vscode.ThemeIcon('browser');
                break;
            case 'function':
                item.iconPath = new vscode.ThemeIcon('symbol-function');
                break;
            case 'app':
                item.iconPath = new vscode.ThemeIcon('package');
                break;
        }
        if (element.command) {
            item.command = {
                command: element.command,
                title: element.label
            };
        }
        return item;
    }
    getChildren(element) {
        if (!element) {
            return Promise.resolve(this.items);
        }
        return Promise.resolve([]);
    }
    async generateComponent(description) {
        try {
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'Generating component...',
                cancellable: false
            }, async (progress) => {
                progress.report({ increment: 0, message: 'Analyzing requirements...' });
                // Get current workspace context
                const context = await this.getWorkspaceContext();
                progress.report({ increment: 30, message: 'Generating code...' });
                // Generate the component code
                const code = await this.aiProvider.generateCode(description, context);
                progress.report({ increment: 70, message: 'Creating file...' });
                // Create new document with generated code
                await this.createNewFile(code, 'component', description);
                progress.report({ increment: 100, message: 'Complete!' });
            });
        }
        catch (error) {
            vscode.window.showErrorMessage(`Failed to generate component: ${error}`);
        }
    }
    async generatePage(description) {
        try {
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'Generating page...',
                cancellable: false
            }, async (progress) => {
                progress.report({ increment: 0, message: 'Planning page structure...' });
                const context = await this.getWorkspaceContext();
                // Enhanced prompt for page generation
                const pagePrompt = `Generate a complete Namel3ss page for: ${description}

Include:
1. Page declaration with appropriate route
2. Layout and styling
3. Data handling if needed
4. Interactive elements
5. Navigation integration

Context: ${context}

Generate clean, production-ready Namel3ss code.`;
                progress.report({ increment: 40, message: 'Generating page code...' });
                const code = await this.aiProvider.generateCode(pagePrompt, context);
                progress.report({ increment: 80, message: 'Creating page file...' });
                await this.createNewFile(code, 'page', description);
                progress.report({ increment: 100, message: 'Page created!' });
            });
        }
        catch (error) {
            vscode.window.showErrorMessage(`Failed to generate page: ${error}`);
        }
    }
    async explainCode(code) {
        try {
            return await this.aiProvider.explainCode(code);
        }
        catch (error) {
            return `Error explaining code: ${error}`;
        }
    }
    async getWorkspaceContext() {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders) {
            return 'No workspace open';
        }
        // Get list of existing N3 files
        const n3Files = await vscode.workspace.findFiles('**/*.n3', null, 10);
        const fileList = n3Files.map(uri => uri.path.split('/').pop()).join(', ');
        // Get current app structure if main app file exists
        const appFiles = await vscode.workspace.findFiles('**/app.n3', null, 1);
        let appStructure = '';
        if (appFiles.length > 0) {
            try {
                const appDoc = await vscode.workspace.openTextDocument(appFiles[0]);
                appStructure = appDoc.getText().substring(0, 500); // First 500 chars
            }
            catch (error) {
                appStructure = 'Could not read app file';
            }
        }
        return `
Workspace: ${workspaceFolders[0].name}
Existing N3 files: ${fileList || 'none'}
App structure: ${appStructure || 'No main app file found'}
`;
    }
    async createNewFile(code, type, description) {
        // Generate filename from description
        const filename = this.generateFilename(description, type);
        // Create new untitled document
        const doc = await vscode.workspace.openTextDocument({
            content: code,
            language: 'namel3ss'
        });
        // Show the document
        const editor = await vscode.window.showTextDocument(doc);
        // Show save dialog with suggested filename
        const saveUri = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.file(filename),
            filters: {
                'Namel3ss Files': ['n3'],
                'All Files': ['*']
            }
        });
        if (saveUri) {
            await vscode.workspace.fs.writeFile(saveUri, Buffer.from(code));
            // Close the untitled document and open the saved file
            await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
            await vscode.window.showTextDocument(saveUri);
            vscode.window.showInformationMessage(`✅ ${type} created: ${saveUri.path.split('/').pop()}`);
        }
    }
    generateFilename(description, type) {
        // Convert description to filename
        const name = description
            .toLowerCase()
            .replace(/[^a-z0-9\s]/g, '')
            .replace(/\s+/g, '_')
            .substring(0, 30);
        const prefix = type === 'page' ? 'page_' : '';
        return `${prefix}${name}.n3`;
    }
    // Quick generation templates
    async generateQuickComponent(type) {
        const templates = {
            dashboard: 'Create a dashboard page with overview widgets, charts, and key metrics',
            form: 'Create a form component with validation, input fields, and submit handling',
            chart: 'Create a chart component with data visualization and interactive features',
            table: 'Create a data table component with sorting, filtering, and pagination'
        };
        await this.generateComponent(templates[type]);
    }
    async generateFromTemplate(templateName) {
        const templates = {
            'crud-page': 'Create a CRUD page with list view, add/edit forms, and delete functionality',
            'auth-flow': 'Create authentication pages with login, register, and password reset',
            'settings-page': 'Create a settings page with tabs, form controls, and save functionality',
            'dashboard-home': 'Create a main dashboard with navigation, widgets, and overview stats'
        };
        const description = templates[templateName];
        if (description) {
            await this.generatePage(description);
        }
    }
    refresh() {
        this._onDidChangeTreeData.fire();
    }
}
exports.CodeGenerationProvider = CodeGenerationProvider;
//# sourceMappingURL=codeGenerationProvider.js.map