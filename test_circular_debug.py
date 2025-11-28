"""Debug circular dependency detection."""

import tempfile
from pathlib import Path
from namel3ss.modules.system import ModuleResolver, CircularDependencyError

# Create temporary files
with tempfile.TemporaryDirectory() as tmpdir:
    project_root = Path(tmpdir)
    (project_root / "myapp").mkdir()
    
    # Create circular dependency: a -> b -> a
    a_file = project_root / "myapp" / "a.ai"
    a_file.write_text('module myapp.a\nimport myapp.b\napp "A" { }')
    
    b_file = project_root / "myapp" / "b.ai"
    b_file.write_text('module myapp.b\nimport myapp.a\napp "B" { }')
    
    print("Files created:")
    print(f"  a.ai: {a_file.read_text()!r}")
    print(f"  b.ai: {b_file.read_text()!r}")
    
    # Patch the resolver to add debug output
    original_load = ModuleResolver.load_module
    def debug_load(self, module_name, source_path=None):
        print(f"\n  load_module({module_name!r})")
        print(f"    loading_stack BEFORE: {self.loading_stack}")
        print(f"    loaded_modules: {list(self.loaded_modules.keys())}")
        print(f"    checking: {module_name} in {self.loaded_modules}")
        result = original_load(self, module_name, source_path)
        print(f"    loading_stack AFTER: {self.loading_stack}")
        return result
    
    ModuleResolver.load_module = debug_load
    
    resolver = ModuleResolver(project_root=str(project_root))
    
    try:
        print("\nLoading myapp.a...")
        module_info = resolver.load_module("myapp.a")
        print(f"\n  Loaded successfully!")
        print(f"  Imports: {module_info.imports}")
        print(f"  All loaded modules: {list(resolver.loaded_modules.keys())}")
        print(f"  Dependency graph: {dict(resolver.dependency_graph)}")
    except CircularDependencyError as e:
        print(f"\n  CircularDependencyError raised: {e}")
    except Exception as e:
        print(f"\n  Other error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

