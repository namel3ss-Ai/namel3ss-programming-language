"""
Integration tests that verify real example files have proper App wiring.
"""
import pytest
from pathlib import Path
from namel3ss.loader import load_program, extract_single_app


# Path to examples directory
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


class TestRealExamples:
    """Test that real example files are properly wired to App."""
    
    def test_memory_chat_demo_wiring(self):
        """Verify memory_chat_demo.n3 has declarations wired to App."""
        file_path = EXAMPLES_DIR / "memory_chat_demo.n3"
        if not file_path.exists():
            pytest.skip(f"Example file not found: {file_path}")
        
        # Load and extract the App
        program = load_program(str(file_path))
        app = extract_single_app(program)
        
        # Verify App has declarations attached
        assert app is not None, "Expected App object"
        assert app.name, "App should have a name"
        
        # Check for expected declarations based on file content
        # memory_chat_demo.n3 has: llms, memories, prompts, chains
        print(f"\nApp '{app.name}' has:")
        print(f"  - {len(app.llms)} LLMs: {[llm.name for llm in app.llms]}")
        print(f"  - {len(app.memories)} memories: {[m.name for m in app.memories]}")
        print(f"  - {len(app.prompts)} prompts: {[p.name for p in app.prompts]}")
        print(f"  - {len(app.chains)} chains: {[c.name for c in app.chains]}")
        
        # These assertions verify the wiring is working
        assert len(app.llms) > 0, "Expected LLMs to be wired to App"
        assert len(app.memories) > 0, "Expected memories to be wired to App"
        # Note: Some examples may not have all declaration types
    
    def test_rag_demo_wiring(self):
        """Verify rag_demo.n3 has declarations wired to App."""
        file_path = EXAMPLES_DIR / "rag_demo.n3"
        if not file_path.exists():
            pytest.skip(f"Example file not found: {file_path}")
        
        program = load_program(str(file_path))
        app = extract_single_app(program)
        
        assert app is not None, "Expected App object"
        assert app.name, "App should have a name"
        
        print(f"\nApp '{app.name}' has:")
        print(f"  - {len(app.datasets)} datasets: {[d.name for d in app.datasets]}")
        print(f"  - {len(app.indices)} indices: {[i.name for i in app.indices]}")
        print(f"  - {len(app.rag_pipelines)} RAG pipelines: {[r.name for r in app.rag_pipelines]}")
        
        # Verify wiring (actual counts depend on file content)
        # At minimum, a RAG demo should have some of these
        total_declarations = (len(app.datasets) + len(app.indices) + 
                            len(app.rag_pipelines) + len(app.llms))
        assert total_declarations > 0, "Expected some declarations wired to App"
    
    def test_provider_demo_wiring(self):
        """Verify provider_demo.n3 has declarations wired to App."""
        file_path = EXAMPLES_DIR / "provider_demo.n3"
        if not file_path.exists():
            pytest.skip(f"Example file not found: {file_path}")
        
        program = load_program(str(file_path))
        app = extract_single_app(program)
        
        assert app is not None, "Expected App object"
        
        print(f"\nApp '{app.name}' has:")
        print(f"  - {len(app.llms)} LLMs: {[llm.name for llm in app.llms]}")
        print(f"  - {len(app.prompts)} prompts: {[p.name for p in app.prompts]}")
        
        # Provider demo should have LLMs
        assert len(app.llms) > 0, "Expected LLMs to be wired to App"
