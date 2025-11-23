/**
 * AI Provider - Unified interface for different AI models
 */

import * as vscode from 'vscode';

export interface AIResponse {
    text: string;
    usage?: {
        promptTokens: number;
        completionTokens: number;
        totalTokens: number;
    };
    model?: string;
}

export interface AIMessage {
    role: 'system' | 'user' | 'assistant';
    content: string;
}

export abstract class BaseAIProvider {
    constructor(protected name: string, protected model: string) {}

    abstract generateCompletion(prompt: string, options?: any): Promise<AIResponse>;
    abstract generateChat(messages: AIMessage[], options?: any): Promise<AIResponse>;
    abstract streamCompletion?(prompt: string, callback: (chunk: string) => void, options?: any): Promise<void>;
}

export class OpenAIProvider extends BaseAIProvider {
    private openai: any;

    constructor(apiKey: string, model: string = 'gpt-4') {
        super('openai', model);
        // In real implementation, would import and initialize OpenAI client
        this.openai = { apiKey, model };
    }

    async generateCompletion(prompt: string, options: any = {}): Promise<AIResponse> {
        // Mock implementation for demo
        return {
            text: `AI response to: ${prompt.substring(0, 50)}...`,
            usage: { promptTokens: 100, completionTokens: 50, totalTokens: 150 },
            model: this.model
        };
    }

    async generateChat(messages: AIMessage[], options: any = {}): Promise<AIResponse> {
        const lastMessage = messages[messages.length - 1];
        return {
            text: `Chat response to: ${lastMessage.content.substring(0, 50)}...`,
            usage: { promptTokens: 120, completionTokens: 60, totalTokens: 180 },
            model: this.model
        };
    }

    async streamCompletion(prompt: string, callback: (chunk: string) => void, options: any = {}): Promise<void> {
        // Mock streaming
        const response = `Streaming response to: ${prompt.substring(0, 30)}...`;
        const chunks = response.split(' ');
        
        for (const chunk of chunks) {
            callback(chunk + ' ');
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }
}

export class AnthropicProvider extends BaseAIProvider {
    constructor(apiKey: string, model: string = 'claude-3-sonnet-20240229') {
        super('anthropic', model);
    }

    async generateCompletion(prompt: string, options: any = {}): Promise<AIResponse> {
        return {
            text: `Claude response: ${prompt.substring(0, 50)}...`,
            model: this.model
        };
    }

    async generateChat(messages: AIMessage[], options: any = {}): Promise<AIResponse> {
        const lastMessage = messages[messages.length - 1];
        return {
            text: `Claude chat: ${lastMessage.content.substring(0, 50)}...`,
            model: this.model
        };
    }
}

export class OllamaProvider extends BaseAIProvider {
    constructor(model: string = 'llama3') {
        super('ollama', model);
    }

    async generateCompletion(prompt: string, options: any = {}): Promise<AIResponse> {
        return {
            text: `Ollama response: ${prompt.substring(0, 50)}...`,
            model: this.model
        };
    }

    async generateChat(messages: AIMessage[], options: any = {}): Promise<AIResponse> {
        const lastMessage = messages[messages.length - 1];
        return {
            text: `Ollama chat: ${lastMessage.content.substring(0, 50)}...`,
            model: this.model
        };
    }
}

export class AiProvider {
    private provider: BaseAIProvider;

    constructor(private context: vscode.ExtensionContext) {
        this.initialize();
    }

    private initialize() {
        const config = vscode.workspace.getConfiguration('namel3ss.ai');
        const providerName = config.get<string>('provider', 'openai');
        const model = config.get<string>('model', 'gpt-4');
        const apiKey = config.get<string>('apiKey', '');

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

    async generateCode(description: string, context?: string): Promise<string> {
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

    async explainCode(code: string): Promise<string> {
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

    async suggestRefactoring(code: string): Promise<any[]> {
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
        } catch {
            return [{
                title: "General Improvement",
                description: response.text.substring(0, 200),
                type: "optimize",
                priority: "medium",
                code: code
            }];
        }
    }

    async generateTests(code: string): Promise<string> {
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

    async generateDocumentation(code: string): Promise<string> {
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

    async getChatResponse(messages: AIMessage[]): Promise<string> {
        const response = await this.provider.generateChat(messages);
        return response.text;
    }

    async streamChatResponse(messages: AIMessage[], callback: (chunk: string) => void): Promise<void> {
        if (this.provider.streamCompletion) {
            const lastMessage = messages[messages.length - 1];
            await this.provider.streamCompletion(lastMessage.content, callback);
        } else {
            const response = await this.provider.generateChat(messages);
            callback(response.text);
        }
    }

    getProviderInfo(): { name: string; model: string } {
        return {
            name: this.provider.name,
            model: (this.provider as any).model
        };
    }
}