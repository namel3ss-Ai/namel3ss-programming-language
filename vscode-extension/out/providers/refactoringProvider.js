"use strict";
/**
 * Refactoring Provider - AI-powered code refactoring and optimization
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
exports.RefactoringProvider = void 0;
const vscode = __importStar(require("vscode"));
class RefactoringProvider {
    constructor(aiProvider) {
        this.aiProvider = aiProvider;
    }
    async provideCodeActions(document, range, context, token) {
        const actions = [];
        const selectedText = document.getText(range);
        if (selectedText.trim().length === 0)
            return actions;
        // Quick fixes for common issues
        const quickFixes = await this.getQuickFixes(selectedText, document, range);
        quickFixes.forEach(fix => {
            const action = new vscode.CodeAction(fix.title, vscode.CodeActionKind.QuickFix);
            action.edit = this.createWorkspaceEdit(document.uri, range, fix.newCode);
            action.diagnostics = context.diagnostics;
            actions.push(action);
        });
        // Refactoring suggestions
        if (selectedText.length > 10) {
            const refactorings = await this.getRefactoringSuggestions(selectedText, range);
            refactorings.forEach(refactor => {
                const kind = this.getCodeActionKind(refactor.type);
                const action = new vscode.CodeAction(refactor.title, kind);
                action.edit = this.createWorkspaceEdit(document.uri, range, refactor.newCode);
                actions.push(action);
            });
        }
        return actions;
    }
    async provideHover(document, position, token) {
        const range = document.getWordRangeAtPosition(position);
        if (!range)
            return null;
        const word = document.getText(range);
        const lineText = document.lineAt(position).text;
        // Provide contextual help for Namel3ss syntax
        const hoverText = this.getHoverText(word, lineText);
        if (hoverText) {
            return new vscode.Hover(new vscode.MarkdownString(hoverText), range);
        }
        return null;
    }
    async getRefactoringSuggestions(code, selection) {
        try {
            const suggestions = await this.aiProvider.suggestRefactoring(code);
            return suggestions.map(s => ({
                ...s,
                range: selection || new vscode.Range(0, 0, 0, code.length)
            }));
        }
        catch (error) {
            console.error('Refactoring error:', error);
            return [];
        }
    }
    async getQuickFixes(code, document, range) {
        const fixes = [];
        // Common quick fixes for Namel3ss
        if (code.includes('show') && !code.includes(':')) {
            fixes.push({
                title: 'Add missing colon after show',
                description: 'Add colon to make valid Namel3ss syntax',
                type: 'fix',
                priority: 'high',
                range,
                newCode: code.replace('show ', 'show: ')
            });
        }
        if (code.includes('"') && (code.split('"').length - 1) % 2 !== 0) {
            fixes.push({
                title: 'Fix unclosed string',
                description: 'Add missing closing quote',
                type: 'fix',
                priority: 'high',
                range,
                newCode: code + '"'
            });
        }
        return fixes;
    }
    async getPerformanceOptimizations(code) {
        try {
            const prompt = `Analyze this Namel3ss code for performance optimizations:\n\n${code}\n\nSuggest specific improvements for better performance, memory usage, or execution speed.`;
            const response = await this.aiProvider.generateCode(prompt);
            // Parse response into suggestions
            return [{
                    title: 'Performance Optimization',
                    description: response,
                    type: 'optimize',
                    priority: 'medium',
                    range: new vscode.Range(0, 0, 0, 0),
                    newCode: code
                }];
        }
        catch (error) {
            return [];
        }
    }
    async applyRefactoring(suggestion, editor) {
        const edit = new vscode.WorkspaceEdit();
        edit.replace(editor.document.uri, suggestion.range, suggestion.newCode);
        await vscode.workspace.applyEdit(edit);
        vscode.window.showInformationMessage(`✅ Applied: ${suggestion.title}`);
    }
    async applyQuickFix(fix, editor) {
        await this.applyRefactoring(fix, editor);
    }
    getOptimizationWebview(suggestions) {
        return `
<!DOCTYPE html>
<html>
<head>
    <title>Performance Optimizations</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 20px; }
        .suggestion { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .title { font-weight: bold; color: #0066cc; }
        .description { margin: 10px 0; }
        .priority { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 12px; }
        .high { background: #ff4444; color: white; }
        .medium { background: #ffaa00; color: white; }
        .low { background: #00aa00; color: white; }
    </style>
</head>
<body>
    <h2>🚀 Performance Optimization Suggestions</h2>
    ${suggestions.map(s => `
        <div class="suggestion">
            <div class="title">${s.title} <span class="priority ${s.priority}">${s.priority.toUpperCase()}</span></div>
            <div class="description">${s.description}</div>
        </div>
    `).join('')}
</body>
</html>`;
    }
    getTreeProvider() {
        return {
            getChildren: () => Promise.resolve([
                { label: 'Extract Component', description: 'Extract reusable component' },
                { label: 'Optimize Performance', description: 'Improve code performance' },
                { label: 'Fix Code Issues', description: 'Auto-fix common problems' }
            ]),
            getTreeItem: (element) => {
                const item = new vscode.TreeItem(element.label);
                item.description = element.description;
                item.iconPath = new vscode.ThemeIcon('gear');
                return item;
            }
        };
    }
    getHoverText(word, lineText) {
        const hoverTexts = {
            'app': '**App Declaration**\n\nDefines the main application with name and configuration.\n\n`app "MyApp" with auth`',
            'page': '**Page Declaration**\n\nDefines a page with route and content.\n\n`page "Home" at "/": show text "Welcome"`',
            'frame': '**Data Frame**\n\nDefines a data structure for type safety.\n\n`frame UserData { name: text, email: text }`',
            'show': '**Show Element**\n\nDisplays UI elements like text, buttons, forms.\n\n`show text "Hello World"`',
            'widget': '**Widget Declaration**\n\nDefines interactive UI components.\n\n`widget toast "Message"`',
            'data': '**Data Source**\n\nDefines where data comes from.\n\n`data from api "/users"`',
            'style': '**Styling**\n\nApplies CSS styles to elements.\n\n`style { color: blue }`'
        };
        return hoverTexts[word] || null;
    }
    getCodeActionKind(type) {
        switch (type) {
            case 'extract': return vscode.CodeActionKind.RefactorExtract;
            case 'inline': return vscode.CodeActionKind.RefactorInline;
            case 'rename': return vscode.CodeActionKind.RefactorRewrite;
            case 'optimize': return vscode.CodeActionKind.RefactorRewrite;
            case 'fix': return vscode.CodeActionKind.QuickFix;
            default: return vscode.CodeActionKind.Refactor;
        }
    }
    createWorkspaceEdit(uri, range, newText) {
        const edit = new vscode.WorkspaceEdit();
        edit.replace(uri, range, newText);
        return edit;
    }
}
exports.RefactoringProvider = RefactoringProvider;
//# sourceMappingURL=refactoringProvider.js.map