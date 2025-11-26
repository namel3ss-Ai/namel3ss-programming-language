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
export declare abstract class BaseAIProvider {
    protected name: string;
    protected model: string;
    constructor(name: string, model: string);
    abstract generateCompletion(prompt: string, options?: any): Promise<AIResponse>;
    abstract generateChat(messages: AIMessage[], options?: any): Promise<AIResponse>;
    abstract streamCompletion?(prompt: string, callback: (chunk: string) => void, options?: any): Promise<void>;
}
export declare class OpenAIProvider extends BaseAIProvider {
    private openai;
    constructor(apiKey: string, model?: string);
    generateCompletion(prompt: string, options?: any): Promise<AIResponse>;
    generateChat(messages: AIMessage[], options?: any): Promise<AIResponse>;
    streamCompletion(prompt: string, callback: (chunk: string) => void, options?: any): Promise<void>;
}
export declare class AnthropicProvider extends BaseAIProvider {
    constructor(apiKey: string, model?: string);
    generateCompletion(prompt: string, options?: any): Promise<AIResponse>;
    generateChat(messages: AIMessage[], options?: any): Promise<AIResponse>;
}
export declare class OllamaProvider extends BaseAIProvider {
    constructor(model?: string);
    generateCompletion(prompt: string, options?: any): Promise<AIResponse>;
    generateChat(messages: AIMessage[], options?: any): Promise<AIResponse>;
}
export declare class AiProvider {
    private context;
    private provider;
    constructor(context: vscode.ExtensionContext);
    private initialize;
    generateCode(description: string, context?: string): Promise<string>;
    explainCode(code: string): Promise<string>;
    suggestRefactoring(code: string): Promise<any[]>;
    generateTests(code: string): Promise<string>;
    generateDocumentation(code: string): Promise<string>;
    getChatResponse(messages: AIMessage[]): Promise<string>;
    streamChatResponse(messages: AIMessage[], callback: (chunk: string) => void): Promise<void>;
    getProviderInfo(): {
        name: string;
        model: string;
    };
}
//# sourceMappingURL=aiProvider.d.ts.map