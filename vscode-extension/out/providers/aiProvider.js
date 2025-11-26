"use strict";
/**
 * AI Provider - Unified interface for different AI models
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
exports.AiProvider = exports.OllamaProvider = exports.AnthropicProvider = exports.OpenAIProvider = exports.BaseAIProvider = void 0;
const vscode = __importStar(require("vscode"));
class BaseAIProvider {
    constructor(name, model) {
        this.name = name;
        this.model = model;
    }
}
exports.BaseAIProvider = BaseAIProvider;
class OpenAIProvider extends BaseAIProvider {
    constructor(apiKey, model = 'gpt-4') {
        super('openai', model);
        // In real implementation, would import and initialize OpenAI client
        this.openai = { apiKey, model };
    }
    async generateCompletion(prompt, options = {}) {
        // Mock implementation for demo
        return {
            text: `AI response to: ${prompt.substring(0, 50)}...`,
            usage: { promptTokens: 100, completionTokens: 50, totalTokens: 150 },
            model: this.model
        };
    }
    async generateChat(messages, options = {}) {
        const lastMessage = messages[messages.length - 1];
        return {
            text: `Chat response to: ${lastMessage.content.substring(0, 50)}...`,
            usage: { promptTokens: 120, completionTokens: 60, totalTokens: 180 },
            model: this.model
        };
    }
    async streamCompletion(prompt, callback, options = {}) {
        // Mock streaming
        const response = `Streaming response to: ${prompt.substring(0, 30)}...`;
        const chunks = response.split(' ');
        for (const chunk of chunks) {
            callback(chunk + ' ');
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }
}
exports.OpenAIProvider = OpenAIProvider;
class AnthropicProvider extends BaseAIProvider {
    constructor(apiKey, model = 'claude-3-sonnet-20240229') {
        super('anthropic', model);
    }
    async generateCompletion(prompt, options = {}) {
        return {
            text: `Claude response: ${prompt.substring(0, 50)}...`,
            model: this.model
        };
    }
    async generateChat(messages, options = {}) {
        const lastMessage = messages[messages.length - 1];
        return {
            text: `Claude chat: ${lastMessage.content.substring(0, 50)}...`,
            model: this.model
        };
    }
}
exports.AnthropicProvider = AnthropicProvider;
class OllamaProvider extends BaseAIProvider {
    constructor(model = 'llama3') {
        super('ollama', model);
    }
    async generateCompletion(prompt, options = {}) {
        return {
            text: `Ollama response: ${prompt.substring(0, 50)}...`,
            model: this.model
        };
    }
    async generateChat(messages, options = {}) {
        const lastMessage = messages[messages.length - 1];
        return {
            text: `Ollama chat: ${lastMessage.content.substring(0, 50)}...`,
            model: this.model
        };
    }
}
exports.OllamaProvider = OllamaProvider;
class AiProvider {
    constructor(context) {
        this.context = context;
        this.initialize();
    }
    initialize() {
        const config = vscode.workspace.getConfiguration('namel3ss.ai');
        const providerName = config.get('provider', 'openai');
        const model = config.get('model', 'gpt-4');
        const apiKey = config.get('apiKey', '');
        switch (providerName) {
            case 'openai':
                this.provider = new OpenAIProvider(apiKey, model);
                break;
            case 'anthropic':
                this.provider = new AnthropicProvider(apiKey, model);
                break;
            case 'ollama':
                this.provider = new OllamaProvider(model);
                break;
            default:
                this.provider = new OpenAIProvider(apiKey, model);
        }
    }
    async generateCode(description, context) {
        const prompt = `
Generate Namel3ss code for the following description:
${description}

${context ? `Context:\n${context}` : ''}

Generate clean, well-structured Namel3ss code with proper syntax and best practices.
Focus on creating functional components with appropriate styling and data handling.
`;
        const response = await this.provider.generateCompletion(prompt, {
            temperature: 0.3,
            maxTokens: 2048
        });
        return response.text;
    }
    async explainCode(code) {
        const prompt = `
Explain the following Namel3ss code in clear, concise terms:

${code}

Provide:
1. What this code does
2. Key components and their purpose  
3. Any notable patterns or techniques used
4. Potential improvements or considerations
`;
        const response = await this.provider.generateCompletion(prompt);
        return response.text;
    }
    async suggestRefactoring(code) {
        const prompt = `
Analyze this Namel3ss code and suggest refactoring improvements:

${code}

Provide suggestions as JSON array with format:
[
  {
    "title": "Brief title",
    "description": "Detailed description", 
    "type": "extract|inline|rename|optimize",
    "priority": "low|medium|high",
    "code": "refactored code"
  }
]
`;
        const response = await this.provider.generateCompletion(prompt);
        try {
            return JSON.parse(response.text);
        }
        catch {
            return [{
                    title: "General Improvement",
                    description: response.text.substring(0, 200),
                    type: "optimize",
                    priority: "medium",
                    code: code
                }];
        }
    }
    async generateTests(code) {
        const prompt = `
Generate comprehensive test cases for this Namel3ss code:

${code}

Create tests that cover:
1. Basic functionality
2. Edge cases
3. Error conditions
4. User interactions (if applicable)

Use standard testing patterns and include setup/teardown as needed.
`;
        const response = await this.provider.generateCompletion(prompt);
        return response.text;
    }
    async generateDocumentation(code) {
        const prompt = `
Generate comprehensive documentation for this Namel3ss code:

${code}

Include:
1. Overview and purpose
2. Component/function descriptions
3. Parameters and return values
4. Usage examples
5. Implementation notes

Use clear, professional documentation style.
`;
        const response = await this.provider.generateCompletion(prompt);
        return response.text;
    }
    async getChatResponse(messages) {
        const response = await this.provider.generateChat(messages);
        return response.text;
    }
    async streamChatResponse(messages, callback) {
        if (this.provider.streamCompletion) {
            const lastMessage = messages[messages.length - 1];
            await this.provider.streamCompletion(lastMessage.content, callback);
        }
        else {
            const response = await this.provider.generateChat(messages);
            callback(response.text);
        }
    }
    getProviderInfo() {
        return {
            name: this.provider.name,
            model: this.provider.model
        };
    }
}
exports.AiProvider = AiProvider;
//# sourceMappingURL=aiProvider.js.map