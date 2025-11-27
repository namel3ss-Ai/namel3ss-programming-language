"""
Quick validation script for complex education platform
Tests parsing of advanced AI features
"""

from namel3ss.parser import Parser
from pathlib import Path

def main():
    print("Testing Advanced AI-Powered Education Platform...")
    print("=" * 60)
    
    # Load and parse
    path = Path("examples") / "education-quiz-suite.ai"
    content = path.read_text(encoding='utf-8')
    
    try:
        parser = Parser(content)
        app = parser.parse_app()
        
        print(f"\n✅ PARSING SUCCESS!\n")
        print(f"App: {app.name}\n")
        
        # AI Components
        print("AI Infrastructure:")
        print(f"  - LLMs: {len(app.llms)}")
        if app.llms:
            print(f"    ({', '.join([llm.name for llm in app.llms])})")
        
        print(f"  - Memory Systems: {len(app.memories)}")
        if app.memories:
            print(f"    ({', '.join([m.name for m in app.memories])})")
        
        print(f"  - Prompts: {len(app.prompts)}")
        if app.prompts:
            print(f"    ({', '.join([p.name for p in app.prompts])})")
        
        print(f"  - Tools: {len(app.tools)}")
        if app.tools:
            print(f"    ({', '.join([t.name for t in app.tools])})")
        
        print(f"  - Agents: {len(app.agents)}")
        if app.agents:
            print(f"    ({', '.join([a.name for a in app.agents])})")
        
        # Data
        print(f"\nData Layer:")
        print(f"  - Datasets: {len(app.datasets)}")
        if app.datasets:
            dataset_names = [d.name for d in app.datasets]
            print(f"    ({', '.join(dataset_names[:10])}{'...' if len(dataset_names) > 10 else ''})")
        
        # UI
        print(f"\nUser Interface:")
        print(f"  - Pages: {len(app.pages)}")
        if app.pages:
            print(f"    ({', '.join([p.name for p in app.pages])})")
        
        print(f"\n✅ All components parsed successfully!")
        print(f"\n🎉 Advanced AI-Powered Education Platform is ready!")
        
    except Exception as e:
        print(f"\n❌ Parsing failed:")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
