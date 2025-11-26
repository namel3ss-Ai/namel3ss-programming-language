"use strict";
/**
 * Completion Provider - AI-powered intelligent code completion
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
exports.CompletionProvider = void 0;
const vscode = __importStar(require("vscode"));
class CompletionProvider {
    constructor(aiProvider) {
        this.aiProvider = aiProvider;
    }
    async provideCompletionItems(document, position, token, context) {
        // Get context around cursor
        const lineText = document.lineAt(position).text;
        const textBefore = document.getText(new vscode.Range(Math.max(0, position.line - 5), 0, position.line, position.character));
        // Basic completions for Namel3ss syntax
        const basicCompletions = this.getBasicCompletions(lineText, position);
        // AI-powered completions for more complex scenarios
        if (this.shouldUseAICompletion(lineText, textBefore)) {
            const aiCompletions = await this.getAICompletions(textBefore, position);
            return [...basicCompletions, ...aiCompletions];
        }
        return basicCompletions;
    }
    getBasicCompletions(lineText, position) {
        const completions = [];
        // App structure completions
        if (lineText.includes('app ') || lineText.startsWith('app')) {
            completions.push(this.createCompletionItem('app "MyApp"', 'App declaration', vscode.CompletionItemKind.Keyword), this.createCompletionItem('app "MyApp" with auth', 'App with authentication', vscode.CompletionItemKind.Snippet));
        }
        // Page completions
        if (lineText.includes('page ') || lineText.startsWith('page')) {
            completions.push(this.createCompletionItem('page "Home" at "/"', 'Basic page', vscode.CompletionItemKind.Keyword), this.createCompletionItem('page "Dashboard" at "/dashboard" with auth', 'Protected page', vscode.CompletionItemKind.Snippet));
        }
        // Frame completions
        if (lineText.includes('frame ') || lineText.startsWith('frame')) {
            completions.push(this.createCompletionItem('frame UserData', 'Data frame', vscode.CompletionItemKind.Struct), this.createCompletionItem('frame UserData { name: text, email: text }', 'Frame with fields', vscode.CompletionItemKind.Snippet));
        }
        // Show element completions
        if (lineText.includes('show ') || lineText.trim() === 'show') {
            completions.push(this.createCompletionItem('show text "Hello"', 'Text element', vscode.CompletionItemKind.Value), this.createCompletionItem('show button "Click me"', 'Button element', vscode.CompletionItemKind.Value), this.createCompletionItem('show form', 'Form element', vscode.CompletionItemKind.Value), this.createCompletionItem('show chart', 'Chart element', vscode.CompletionItemKind.Value), this.createCompletionItem('show table', 'Table element', vscode.CompletionItemKind.Value));
        }
        // Widget completions
        if (lineText.includes('widget ') || lineText.startsWith('widget')) {
            completions.push(this.createCompletionItem('widget toast "Message"', 'Toast notification', vscode.CompletionItemKind.Function), this.createCompletionItem('widget modal "Title"', 'Modal dialog', vscode.CompletionItemKind.Function));
        }
        // Data completions
        if (lineText.includes('data ') || lineText.startsWith('data')) {
            completions.push(this.createCompletionItem('data from api "/users"', 'API data source', vscode.CompletionItemKind.Variable), this.createCompletionItem('data from static', 'Static data', vscode.CompletionItemKind.Variable), this.createCompletionItem('data from file', 'File data source', vscode.CompletionItemKind.Variable));
        }
        // Style completions
        if (lineText.includes('style ') || lineText.trim() === 'style') {
            completions.push(this.createCompletionItem('style { color: blue }', 'Inline styles', vscode.CompletionItemKind.Property), this.createCompletionItem('style class "my-class"', 'CSS class', vscode.CompletionItemKind.Property));
        }
        // Function completions
        if (lineText.includes('fn ') || lineText.startsWith('fn')) {
            completions.push(this.createCompletionItem('fn myFunction()', 'Function declaration', vscode.CompletionItemKind.Function), this.createCompletionItem('fn myFunction(param: type)', 'Function with parameter', vscode.CompletionItemKind.Snippet));
        }
        return completions;
    }
    async getAICompletions(textBefore, position) {
        try {
            const prompt = `
Given this Namel3ss code context:

${textBefore}

Suggest 3-5 intelligent code completions that would make sense at this position. Consider:
1. The current context and what the user might want to add next
2. Common patterns in Namel3ss
3. Best practices for the current component type

Return suggestions as JSON array:
[
  {
    "text": "completion text",
    "description": "what this does",
    "type": "keyword|snippet|property|value"
  }
]
`;
            const response = await this.aiProvider.generateCode(prompt);
            try {
                const suggestions = JSON.parse(response);
                return suggestions.map((s) => {
                    const kind = this.getCompletionKind(s.type);
                    return this.createCompletionItem(s.text, s.description, kind);
                });
            }
            catch {
                // Fallback to single completion
                return [
                    this.createCompletionItem(response.substring(0, 50), 'AI suggestion', vscode.CompletionItemKind.Text)
                ];
            }
        }
        catch (error) {
            console.error('AI completion error:', error);
            return [];
        }
    }
    shouldUseAICompletion(lineText, textBefore) {
        // Use AI for more complex completions
        return (textBefore.length > 100 || // Substantial context
            lineText.includes('{') || // Object/style context
            lineText.includes('data') || // Data context
            lineText.includes('with') || // Advanced syntax
            lineText.trim().length === 0 // Empty line - suggest next step
        );
    }
    createCompletionItem(text, description, kind) {
        const item = new vscode.CompletionItem(text, kind);
        item.detail = description;
        item.documentation = new vscode.MarkdownString(`**${description}**\\n\\nInserts: \`${text}\``);
        // Set insert text with proper formatting
        if (text.includes('\\n') || text.includes('{')) {
            item.insertText = new vscode.SnippetString(text);
        }
        else {
            item.insertText = text;
        }
        // Sort order - prioritize based on relevance
        switch (kind) {
            case vscode.CompletionItemKind.Keyword:
                item.sortText = 'a' + text;
                break;
            case vscode.CompletionItemKind.Snippet:
                item.sortText = 'b' + text;
                break;
            default:
                item.sortText = 'c' + text;
        }
        return item;
    }
    getCompletionKind(type) {
        switch (type) {
            case 'keyword': return vscode.CompletionItemKind.Keyword;
            case 'snippet': return vscode.CompletionItemKind.Snippet;
            case 'property': return vscode.CompletionItemKind.Property;
            case 'value': return vscode.CompletionItemKind.Value;
            case 'function': return vscode.CompletionItemKind.Function;
            default: return vscode.CompletionItemKind.Text;
        }
    }
}
exports.CompletionProvider = CompletionProvider;
//# sourceMappingURL=completionProvider.js.map