"use strict";
/**
 * Chat Provider - AI Chat Interface for VS Code
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
exports.ChatProvider = void 0;
const vscode = __importStar(require("vscode"));
class ChatProvider {
    constructor(context, aiProvider) {
        this.context = context;
        this.aiProvider = aiProvider;
        this._messages = [];
    }
    resolveWebviewView(webviewView, context, _token) {
        this._view = webviewView;
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this.context.extensionUri]
        };
        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);
        // Handle messages from webview
        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'sendMessage':
                    await this.handleUserMessage(data.message);
                    break;
                case 'clearChat':
                    this.clearChat();
                    break;
                case 'exportChat':
                    this.exportChat();
                    break;
            }
        });
    }
    async openChat() {
        if (this._view) {
            this._view.show?.(true);
        }
        else {
            // Open the view if it's not already open
            await vscode.commands.executeCommand('namel3ssChat.focus');
        }
    }
    async handleUserMessage(message) {
        if (!this._view || !message.trim())
            return;
        // Add user message
        this._messages.push({ role: 'user', content: message });
        this.addMessageToChat('user', message);
        // Show typing indicator
        this.showTypingIndicator(true);
        try {
            // Get context from current file if available
            const context = await this.getCurrentContext();
            // Prepare messages with system context
            const messagesWithContext = [
                {
                    role: 'system',
                    content: `You are an AI assistant for the Namel3ss programming language. You help developers write better Namel3ss code, understand concepts, and solve problems.
                    
                    Current context: ${context}
                    
                    Provide helpful, accurate responses about Namel3ss development. Be concise but informative.`
                },
                ...this._messages
            ];
            // Get AI response with streaming
            let fullResponse = '';
            const messageId = this.addMessageToChat('assistant', '');
            await this.aiProvider.streamChatResponse(messagesWithContext, (chunk) => {
                fullResponse += chunk;
                this.updateMessage(messageId, fullResponse);
            });
            // Add AI response to message history
            this._messages.push({ role: 'assistant', content: fullResponse });
        }
        catch (error) {
            console.error('Chat error:', error);
            this.addMessageToChat('assistant', '❌ Sorry, I encountered an error. Please try again.');
        }
        finally {
            this.showTypingIndicator(false);
        }
    }
    async getCurrentContext() {
        const editor = vscode.window.activeTextEditor;
        if (!editor)
            return 'No file open';
        const document = editor.document;
        if (document.languageId !== 'namel3ss') {
            return `Current file: ${document.fileName} (${document.languageId})`;
        }
        // Get selected text or current line
        const selection = editor.selection;
        let contextText = '';
        if (!selection.isEmpty) {
            contextText = document.getText(selection);
        }
        else {
            const line = document.lineAt(selection.active.line);
            contextText = line.text;
        }
        return `Namel3ss file: ${document.fileName}
Current selection/line: ${contextText}
Total lines: ${document.lineCount}`;
    }
    addMessageToChat(role, content) {
        const messageId = `msg-${Date.now()}-${Math.random()}`;
        if (this._view) {
            this._view.webview.postMessage({
                type: 'addMessage',
                messageId,
                role,
                content,
                timestamp: new Date().toLocaleTimeString()
            });
        }
        return messageId;
    }
    updateMessage(messageId, content) {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'updateMessage',
                messageId,
                content
            });
        }
    }
    showTypingIndicator(show) {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'showTyping',
                show
            });
        }
    }
    clearChat() {
        this._messages = [];
        if (this._view) {
            this._view.webview.postMessage({ type: 'clearMessages' });
        }
    }
    async exportChat() {
        if (this._messages.length === 0) {
            vscode.window.showInformationMessage('No chat messages to export.');
            return;
        }
        const chatContent = this._messages.map(msg => `**${msg.role.toUpperCase()}**: ${msg.content}`).join('\\n\\n');
        const doc = await vscode.workspace.openTextDocument({
            content: chatContent,
            language: 'markdown'
        });
        await vscode.window.showTextDocument(doc);
    }
    _getHtmlForWebview(webview) {
        const scriptNonce = getNonce();
        const styleNonce = getNonce();
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'nonce-${styleNonce}'; script-src 'nonce-${scriptNonce}';">
    <title>Namel3ss AI Chat</title>
    <style nonce="${styleNonce}">
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 10px;
            background: var(--vscode-editor-background);
            color: var(--vscode-editor-foreground);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 10px 0;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .message {
            padding: 10px;
            border-radius: 8px;
            max-width: 85%;
            word-wrap: break-word;
            line-height: 1.4;
        }

        .message.user {
            background: var(--vscode-inputOption-activeBackground);
            align-self: flex-end;
            margin-left: auto;
        }

        .message.assistant {
            background: var(--vscode-textCodeBlock-background);
            align-self: flex-start;
        }

        .message.typing {
            background: var(--vscode-textCodeBlock-background);
            align-self: flex-start;
            font-style: italic;
            opacity: 0.7;
        }

        .message-header {
            font-size: 0.8em;
            opacity: 0.7;
            margin-bottom: 5px;
        }

        .input-container {
            display: flex;
            gap: 10px;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid var(--vscode-panel-border);
        }

        .message-input {
            flex: 1;
            padding: 8px;
            border: 1px solid var(--vscode-input-border);
            border-radius: 4px;
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            outline: none;
            resize: none;
            min-height: 20px;
            max-height: 100px;
        }

        .message-input:focus {
            border-color: var(--vscode-inputOption-activeBorder);
        }

        .send-button, .clear-button {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            cursor: pointer;
            font-size: 14px;
        }

        .send-button:hover, .clear-button:hover {
            background: var(--vscode-button-hoverBackground);
        }

        .clear-button {
            background: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
        }

        .chat-header {
            padding: 10px 0;
            border-bottom: 1px solid var(--vscode-panel-border);
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .chat-title {
            font-weight: bold;
            font-size: 1.1em;
        }

        .export-button {
            padding: 4px 8px;
            font-size: 12px;
            background: transparent;
            border: 1px solid var(--vscode-button-border);
            border-radius: 3px;
            color: var(--vscode-button-foreground);
            cursor: pointer;
        }

        pre, code {
            background: var(--vscode-textPreformat-background);
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }

        pre {
            padding: 10px;
            overflow-x: auto;
            margin: 8px 0;
        }
    </style>
</head>
<body>
    <div class="chat-header">
        <div class="chat-title">🤖 Namel3ss AI Assistant</div>
        <button class="export-button" id="exportBtn">Export</button>
    </div>
    
    <div class="chat-container" id="chatContainer">
        <div class="message assistant">
            <div class="message-header">AI Assistant</div>
            Hi! I'm your Namel3ss AI assistant. I can help you write code, explain concepts, and solve problems. How can I help you today?
        </div>
    </div>
    
    <div class="input-container">
        <textarea 
            class="message-input" 
            id="messageInput" 
            placeholder="Ask me anything about Namel3ss..."
            rows="1"
        ></textarea>
        <button class="send-button" id="sendBtn">Send</button>
        <button class="clear-button" id="clearBtn">Clear</button>
    </div>

    <script nonce="${scriptNonce}">
        const vscode = acquireVsCodeApi();
        const chatContainer = document.getElementById('chatContainer');
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        const clearBtn = document.getElementById('clearBtn');
        const exportBtn = document.getElementById('exportBtn');

        // Auto-resize textarea
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 100) + 'px';
        });

        // Send message
        function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;

            vscode.postMessage({
                type: 'sendMessage',
                message: message
            });

            messageInput.value = '';
            messageInput.style.height = 'auto';
        }

        sendBtn.addEventListener('click', sendMessage);

        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        clearBtn.addEventListener('click', () => {
            vscode.postMessage({ type: 'clearChat' });
        });

        exportBtn.addEventListener('click', () => {
            vscode.postMessage({ type: 'exportChat' });
        });

        // Handle messages from extension
        window.addEventListener('message', event => {
            const message = event.data;
            
            switch (message.type) {
                case 'addMessage':
                    addMessage(message.messageId, message.role, message.content, message.timestamp);
                    break;
                case 'updateMessage':
                    updateMessage(message.messageId, message.content);
                    break;
                case 'showTyping':
                    showTypingIndicator(message.show);
                    break;
                case 'clearMessages':
                    clearMessages();
                    break;
            }
        });

        function addMessage(messageId, role, content, timestamp) {
            const messageDiv = document.createElement('div');
            messageDiv.className = \`message \${role}\`;
            messageDiv.id = messageId;
            
            const headerDiv = document.createElement('div');
            headerDiv.className = 'message-header';
            headerDiv.textContent = \`\${role === 'user' ? 'You' : 'AI Assistant'} - \${timestamp}\`;
            
            const contentDiv = document.createElement('div');
            contentDiv.innerHTML = formatMessage(content);
            
            messageDiv.appendChild(headerDiv);
            messageDiv.appendChild(contentDiv);
            
            // Remove typing indicator if present
            const typingIndicator = chatContainer.querySelector('.message.typing');
            if (typingIndicator) {
                typingIndicator.remove();
            }
            
            chatContainer.appendChild(messageDiv);
            scrollToBottom();
        }

        function updateMessage(messageId, content) {
            const messageElement = document.getElementById(messageId);
            if (messageElement) {
                const contentDiv = messageElement.querySelector('div:last-child');
                if (contentDiv) {
                    contentDiv.innerHTML = formatMessage(content);
                }
                scrollToBottom();
            }
        }

        function showTypingIndicator(show) {
            const existingIndicator = chatContainer.querySelector('.message.typing');
            if (existingIndicator) {
                existingIndicator.remove();
            }

            if (show) {
                const typingDiv = document.createElement('div');
                typingDiv.className = 'message typing';
                typingDiv.innerHTML = '<div class="message-header">AI Assistant</div><div>Thinking...</div>';
                chatContainer.appendChild(typingDiv);
                scrollToBottom();
            }
        }

        function clearMessages() {
            // Keep only the initial welcome message
            const messages = chatContainer.querySelectorAll('.message');
            for (let i = 1; i < messages.length; i++) {
                messages[i].remove();
            }
        }

        function formatMessage(content) {
            // Basic markdown-like formatting
            return content
                .replace(/\`\`\`([\\s\\S]*?)\`\`\`/g, '<pre><code>$1</code></pre>')
                .replace(/\`([^\`]+)\`/g, '<code>$1</code>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\\n/g, '<br>');
        }

        function scrollToBottom() {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        // Focus input on load
        messageInput.focus();
    </script>
</body>
</html>`;
    }
}
exports.ChatProvider = ChatProvider;
ChatProvider.viewType = 'namel3ssChat';
function getNonce() {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}
//# sourceMappingURL=chatProvider.js.map