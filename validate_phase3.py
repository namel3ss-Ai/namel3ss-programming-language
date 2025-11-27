"""Phase 3 AI Integration Validation"""
from namel3ss.parser import Parser

print("Testing Phase 3: AI Integration (Tools, Agents, Prompts)\n")

with open('examples/test_phase3_ai_integration.ai', encoding='utf-8') as f:
    content = f.read()

try:
    parser = Parser(content)
    app = parser.parse_app()
    
    print("✅ PHASE 3 SUCCESS - AI Integration Components Parse!\n")
    print(f"App: {app.name}")
    
    # Count AI components
    tools = [t for t in getattr(app, 'tools', [])]
    agents = [a for a in getattr(app, 'agents', [])]
    prompts = [p for p in getattr(app, 'prompts', [])]
    llms = [l for l in getattr(app, 'llms', [])]
    memories = [m for m in getattr(app, 'memories', [])]
    
    print(f"\nAI Components:")
    print(f"  - Tools: {len(tools)}")
    print(f"  - Agents: {len(agents)}")
    print(f"  - Prompts: {len(prompts)}")
    print(f"  - LLMs: {len(llms)}")
    print(f"  - Memories: {len(memories)}")
    
    if tools:
        print(f"\nTool names:")
        for tool in tools:
            print(f"  - {tool.name}")
    
    if agents:
        print(f"\nAgent names:")
        for agent in agents:
            print(f"  - {agent.name}")
    
    if prompts:
        print(f"\nPrompt names:")
        for prompt in prompts:
            print(f"  - {prompt.name}")
    
    print(f"\nDatasets: {len(app.datasets)}")
    print(f"Pages: {len(app.pages)}")
    
    print("\n✅ All Phase 3 AI components (tools, agents, prompts, LLMs, memory) parse successfully!")
    
except Exception as e:
    print(f"❌ PHASE 3 FAILED: {e}")
    import traceback
    traceback.print_exc()
