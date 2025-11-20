"""
Integration test - Build a simple app and generate backend state.

This tests the full pipeline using refactored code.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_full_integration():
    """Test creating an App and building backend state."""
    print("=" * 60)
    print("INTEGRATION TEST: Full App → Backend State Pipeline")
    print("=" * 60)
    
    try:
        from namel3ss.ast import App, Frame, FrameColumn, Page, Dataset
        from namel3ss.codegen.backend.state import build_backend_state
        
        print("\n1. Creating test App...")
        
        # Create a minimal App
        app = App(
            name="TestApp",
            database="test.db",
            frames=[],
            datasets=[],
            pages=[],
            variables=[],
            prompts=[],
            chains=[],
            connectors=[],
            ai_models=[],
            llms=[],
            tools=[],
            indices=[],
            rag_pipelines=[],
            memories=[],
            insights=[],
            models=[],
            templates=[],
            agents=[],
            graphs=[],
            experiments=[],
            training_jobs=[],
            tuning_jobs=[],
            crud_resources=[],
            evaluators=[],
            metrics=[],
            guardrails=[],
            eval_suites=[],
            queries=[],
            knowledge_modules=[],
        )
        print("✅ Created minimal App")
        
        print("\n2. Building backend state...")
        backend_state = build_backend_state(app)
        print("✅ Backend state built successfully")
        
        print("\n3. Validating backend state structure...")
        assert hasattr(backend_state, 'app'), "Missing 'app' field"
        assert hasattr(backend_state, 'datasets'), "Missing 'datasets' field"
        assert hasattr(backend_state, 'frames'), "Missing 'frames' field"
        assert hasattr(backend_state, 'pages'), "Missing 'pages' field"
        assert hasattr(backend_state, 'env_keys'), "Missing 'env_keys' field"
        print("✅ All required fields present")
        
        print("\n4. Checking backend state content...")
        assert backend_state.app['name'] == "TestApp"
        assert backend_state.app['database'] == "test.db"
        assert isinstance(backend_state.datasets, dict)
        assert isinstance(backend_state.frames, dict)
        assert isinstance(backend_state.pages, list)
        assert isinstance(backend_state.env_keys, list)
        print("✅ Backend state content is valid")
        
        print("\n5. Checking all resource collections...")
        resource_fields = [
            'datasets', 'frames', 'connectors', 'ai_connectors', 'ai_models',
            'llms', 'tools', 'indices', 'rag_pipelines', 'memories', 'prompts',
            'insights', 'models', 'templates', 'chains', 'agents', 'graphs',
            'experiments', 'training_jobs', 'tuning_jobs', 'crud_resources',
            'evaluators', 'metrics', 'guardrails', 'eval_suites', 'queries',
            'knowledge_modules'
        ]
        
        for field in resource_fields:
            assert hasattr(backend_state, field), f"Missing field: {field}"
        print(f"✅ All {len(resource_fields)} resource collections present")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_frame():
    """Test with a Frame to ensure encoding works."""
    print("\n" + "=" * 60)
    print("INTEGRATION TEST: App with Frame")
    print("=" * 60)
    
    try:
        from namel3ss.ast import App, Frame, FrameColumn
        from namel3ss.codegen.backend.state import build_backend_state
        
        print("\n1. Creating app with Frame...")
        
        # Create a frame with columns
        test_frame = Frame(
            name="users",
            columns=[
                FrameColumn(name="id", dtype="integer"),
                FrameColumn(name="name", dtype="string"),
                FrameColumn(name="email", dtype="string"),
            ],
            indexes=[],
            relationships=[],
            constraints=[],
            access=None,
            source=None,
            tags=[],
            metadata={},
            options={},
        )
        
        app = App(
            name="FrameTestApp",
            database="test.db",
            frames=[test_frame],
            datasets=[],
            pages=[],
            variables=[],
            prompts=[],
            chains=[],
            connectors=[],
            ai_models=[],
            llms=[],
            tools=[],
            indices=[],
            rag_pipelines=[],
            memories=[],
            insights=[],
            models=[],
            templates=[],
            agents=[],
            graphs=[],
            experiments=[],
            training_jobs=[],
            tuning_jobs=[],
            crud_resources=[],
            evaluators=[],
            metrics=[],
            guardrails=[],
            eval_suites=[],
            queries=[],
            knowledge_modules=[],
        )
        print("✅ Created app with Frame")
        
        print("\n2. Building backend state with Frame...")
        backend_state = build_backend_state(app)
        print("✅ Backend state built successfully")
        
        print("\n3. Validating Frame encoding...")
        assert 'users' in backend_state.frames, "Frame not in backend state"
        frame_data = backend_state.frames['users']
        assert frame_data['name'] == 'users', "Frame name mismatch"
        assert len(frame_data['columns']) == 3, "Column count mismatch"
        print("✅ Frame encoded correctly")
        print(f"   - Frame: {frame_data['name']}")
        print(f"   - Columns: {len(frame_data['columns'])}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Frame integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("REFACTORING INTEGRATION TESTS")
    print("=" * 60)
    print("Testing real-world usage of refactored packages")
    print("=" * 60 + "\n")
    
    tests = [
        ("Full Pipeline", test_full_integration),
        ("Frame Encoding", test_with_frame),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} integration tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Refactored code works correctly in real-world scenarios!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} integration test(s) failed.")
        sys.exit(1)
