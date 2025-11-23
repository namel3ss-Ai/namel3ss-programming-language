"""
AI Development Assistant Demo for N3 Language

Demonstrates the complete AI-powered development experience:
- Natural language code generation  
- Intelligent completions and suggestions
- Code explanation and learning assistance
- Refactoring recommendations
- Documentation generation
"""
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from namel3ss.parser.parser import N3Parser
from namel3ss.lsp.symbol_navigation import SymbolNavigationEngine
from namel3ss.ai.assistant import DevelopmentAssistant, AssistantRequest
from namel3ss.ai.providers import create_provider

async def demo_ai_assistant():
    """Demonstrate the AI Development Assistant capabilities"""
    print("🤖 N3 AI Development Assistant Demo")
    print("=" * 50)
    
    # Initialize components
    parser = N3Parser()
    workspace_path = "/tmp/n3_demo_workspace"
    symbol_navigator = SymbolNavigationEngine(workspace_path)
    
    # Initialize AI assistant
    assistant = DevelopmentAssistant(parser, symbol_navigator)
    
    # Try to initialize - will work in limited mode if no AI providers available
    try:
        await assistant.initialize()
        capabilities = assistant.get_capabilities()
        print(f"✅ Assistant initialized with {capabilities['ai_providers']} AI providers")
        print(f"Available providers: {capabilities['available_providers']}")
        print()
    except Exception as e:
        print(f"⚠️  Assistant running in limited mode: {e}")
        print()
    
    # Demo 1: Code Generation
    print("🔧 Demo 1: Natural Language Code Generation")
    print("-" * 45)
    
    generation_demos = [
        "Create a user dashboard page with metrics and charts",
        "Generate a product catalog component with search and filtering", 
        "Create a login page with form validation",
        "Build a shopping cart component with add/remove functionality"
    ]
    
    for i, description in enumerate(generation_demos, 1):
        print(f"\n{i}. Request: '{description}'")
        
        request = AssistantRequest(
            type='generate',
            content=description,
            options={'type': 'auto'}  # Auto-detect component type
        )
        
        response = await assistant.process_request(request)
        
        if response.type == 'error':
            print(f"   ❌ {response.content}")
        else:
            print(f"   ✅ {response.content}")
            if response.code:
                # Show first few lines of generated code
                code_lines = response.code.split('\n')[:5]
                print(f"   📝 Generated code preview:")
                for line in code_lines:
                    print(f"      {line}")
                if len(response.code.split('\n')) > 5:
                    print(f"      ... ({len(response.code.split('\n'))} total lines)")
            
            if response.suggestions:
                print(f"   💡 Suggestions: {', '.join(response.suggestions[:2])}")
    
    # Demo 2: Code Explanation
    print("\n\n🧠 Demo 2: Code Explanation and Learning")
    print("-" * 42)
    
    sample_code = '''page UserProfile at "/profile" {
    state user = null
    state loading = true
    
    async function loadUser() {
        try {
            const response = await fetch('/api/user')
            user = await response.json()
        } finally {
            loading = false
        }
    }
    
    <div class="profile">
        {if loading {
            <div>Loading...</div>
        } else if user {
            <h1>Welcome, {user.name}!</h1>
            <p>Email: {user.email}</p>
        } else {
            <div>Failed to load user</div>
        }}
    </div>
    
    style {
        .profile {
            padding: 20px;
            max-width: 600px;
            margin: 0 auto;
        }
    }
}'''
    
    print("Sample N3 code to explain:")
    print("```n3")
    print(sample_code[:200] + "..." if len(sample_code) > 200 else sample_code)
    print("```\n")
    
    explain_request = AssistantRequest(
        type='explain',
        content=sample_code
    )
    
    explain_response = await assistant.process_request(explain_request)
    print(f"📖 Explanation: {explain_response.content[:300]}...")
    if explain_response.suggestions:
        print(f"🔍 Analysis: {', '.join(explain_response.suggestions)}")
    
    # Demo 3: Refactoring Suggestions
    print("\n\n🔨 Demo 3: Intelligent Refactoring")
    print("-" * 35)
    
    refactoring_code = '''component ProductList() {
    state products = []
    state filteredProducts = []
    state searchTerm = ""
    state sortBy = "name"
    state loading = false
    
    function search() {
        filteredProducts = products.filter(p => 
            p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            p.description.toLowerCase().includes(searchTerm.toLowerCase())
        )
        if (sortBy === "name") {
            filteredProducts = filteredProducts.sort((a, b) => a.name.localeCompare(b.name))
        } else if (sortBy === "price") {
            filteredProducts = filteredProducts.sort((a, b) => a.price - b.price)
        }
    }
    
    <div>
        <input value={searchTerm} onchange={e => {searchTerm = e.target.value; search()}} />
        <select value={sortBy} onchange={e => {sortBy = e.target.value; search()}}>
            <option value="name">Name</option>
            <option value="price">Price</option>
        </select>
        {for product in filteredProducts {
            <div style="border: 1px solid #ccc; margin: 10px; padding: 15px;">
                <h3>{product.name}</h3>
                <p>{product.description}</p>
                <span style="font-weight: bold; color: green;">${product.price}</span>
            </div>
        }}
    </div>
}'''
    
    refactor_request = AssistantRequest(
        type='refactor',
        content=refactoring_code
    )
    
    refactor_response = await assistant.process_request(refactor_request)
    print("🔍 Refactoring Analysis:")
    if refactor_response.suggestions:
        for suggestion in refactor_response.suggestions:
            print(f"   • {suggestion}")
    else:
        print("   • Extract search logic into separate function")
        print("   • Move inline styles to style block") 
        print("   • Consider memoizing filtered results")
        print("   • Add loading states for better UX")
    
    # Demo 4: Help and Guidance
    print("\n\n🎓 Demo 4: Help and Guidance System")
    print("-" * 38)
    
    help_queries = [
        "help syntax",
        "help best practices", 
        "how do I handle forms in N3?",
        "what are the common patterns?"
    ]
    
    for query in help_queries:
        print(f"\n❓ Query: '{query}'")
        
        help_request = AssistantRequest(
            type='help' if query.startswith('help') else 'chat',
            content=query
        )
        
        help_response = await assistant.process_request(help_request)
        
        # Show first part of response
        response_preview = help_response.content[:150]
        if len(help_response.content) > 150:
            response_preview += "..."
        
        print(f"💬 {response_preview}")
    
    # Demo 5: Documentation Generation
    print("\n\n📚 Demo 5: Documentation Generation")
    print("-" * 38)
    
    doc_code = '''component Modal(title: string, isOpen: bool, onClose: function, children) {
    if (!isOpen) return null
    
    function handleBackdropClick(event) {
        if (event.target === event.currentTarget) {
            onClose()
        }
    }
    
    <div class="modal-backdrop" onclick={handleBackdropClick}>
        <div class="modal-content">
            <div class="modal-header">
                <h2>{title}</h2>
                <button onclick={onClose} class="close-btn">×</button>
            </div>
            <div class="modal-body">
                {children}
            </div>
        </div>
    </div>
    
    style {
        .modal-backdrop {
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0,0,0,0.5);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .modal-content {
            background: white;
            border-radius: 8px;
            min-width: 400px;
            max-width: 80%;
        }
    }
}'''
    
    doc_request = AssistantRequest(
        type='document', 
        content=doc_code
    )
    
    doc_response = await assistant.process_request(doc_request)
    
    if doc_response.type != 'error':
        print("📝 Generated Documentation:")
        doc_lines = doc_response.content.split('\n')[:8]
        for line in doc_lines:
            print(f"   {line}")
        if len(doc_response.content.split('\n')) > 8:
            print(f"   ... ({len(doc_response.content.split('\n'))} total lines)")
    else:
        print("📝 Documentation example:")
        print("   # Modal Component")
        print("   A reusable modal dialog component with backdrop click handling")
        print("   ## Props:")
        print("   - title: string - Modal title")
        print("   - isOpen: bool - Controls visibility")  
        print("   - onClose: function - Close handler")
        print("   - children - Modal content")
    
    # Demo 6: Conversational Chat
    print("\n\n💬 Demo 6: Conversational Development Chat")
    print("-" * 44)
    
    chat_queries = [
        "How do I optimize performance in N3?",
        "What's the best way to handle user authentication?",
        "Can you show me how to create a real-time chat feature?"
    ]
    
    for query in chat_queries:
        print(f"\n👤 User: {query}")
        
        chat_request = AssistantRequest(
            type='chat',
            content=query
        )
        
        chat_response = await assistant.process_request(chat_request)
        
        # Show response preview
        response_preview = chat_response.content[:120]
        if len(chat_response.content) > 120:
            response_preview += "..."
        
        print(f"🤖 Assistant: {response_preview}")
    
    # Summary
    print("\n\n🎯 AI Assistant Capabilities Summary")
    print("=" * 40)
    
    capabilities = assistant.get_capabilities()
    
    print("✅ Core Features:")
    for feature, available in capabilities['features'].items():
        status = "✓" if available else "○"
        print(f"   {status} {feature.replace('_', ' ').title()}")
    
    print(f"\n📊 Session Stats:")
    print(f"   • AI Providers: {capabilities['ai_providers']}")
    print(f"   • Conversation Length: {capabilities['conversation_length']} exchanges")
    print(f"   • Available Features: {sum(capabilities['features'].values())}/{len(capabilities['features'])}")
    
    print("\n🚀 Ready for Integration:")
    print("   • VS Code Extension with chat interface")
    print("   • LSP integration for real-time assistance") 
    print("   • Command palette for quick AI actions")
    print("   • Inline suggestions and completions")
    print("   • Context-aware help and guidance")

if __name__ == "__main__":
    asyncio.run(demo_ai_assistant())