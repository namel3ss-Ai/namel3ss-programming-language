"""
Direct RLHF API Module Test - Test API modules without torch dependencies.

This test imports API modules directly to avoid torch dependency.
"""

print("=" * 80)
print("RLHF API Module Direct Test")
print("=" * 80)

def test_models_direct():
    """Test models module directly."""
    print("\n[TEST] Models Module (Direct)")
    try:
        import sys
        import importlib.util
        
        # Load models module directly
        spec = importlib.util.spec_from_file_location(
            "models",
            "/Users/disanssebowabasalidde/Documents/GitHub/namel3ss-programming-language/namel3ss/ml/rlhf/models.py"
        )
        models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models)
        
        # Test classes exist
        assert hasattr(models, 'Feedback')
        assert hasattr(models, 'AnnotationTask')
        assert hasattr(models, 'Dataset')
        assert hasattr(models, 'TrainingJob')
        assert hasattr(models, 'FeedbackType')
        
        print("✅ Models module loaded successfully")
        print(f"   - Feedback, AnnotationTask, Dataset, TrainingJob")
        print(f"   - FeedbackType, TaskStatus, DatasetStatus, JobStatus")
        
        # Test creating instances
        feedback = models.Feedback(
            feedback_type=models.FeedbackType.PREFERENCE,
            prompt="Test",
            response_chosen="A",
            response_rejected="B",
        )
        print(f"   - Created Feedback instance: {feedback.feedback_type.value}")
        
        return True
    except Exception as e:
        print(f"❌ Models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_direct():
    """Test database module directly."""
    print("\n[TEST] Database Module (Direct)")
    try:
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "database",
            "/Users/disanssebowabasalidde/Documents/GitHub/namel3ss-programming-language/namel3ss/ml/rlhf/database.py"
        )
        database = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(database)
        
        assert hasattr(database, 'DatabaseManager')
        assert hasattr(database, 'initialize_database')
        assert hasattr(database, 'get_database')
        assert hasattr(database, 'get_session')
        
        print("✅ Database module loaded successfully")
        print(f"   - DatabaseManager, initialize_database")
        print(f"   - get_database, get_session")
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_schemas_direct():
    """Test schemas module directly."""
    print("\n[TEST] Schemas Module (Direct)")
    try:
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "schemas",
            "/Users/disanssebowabasalidde/Documents/GitHub/namel3ss-programming-language/namel3ss/ml/rlhf/schemas.py"
        )
        schemas = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(schemas)
        
        assert hasattr(schemas, 'PreferenceFeedbackCreate')
        assert hasattr(schemas, 'TaskCreate')
        assert hasattr(schemas, 'DatasetCreate')
        assert hasattr(schemas, 'JobCreate')
        
        print("✅ Schemas module loaded successfully")
        print(f"   - Feedback, Task, Dataset, Job schemas")
        
        # Test creating schema instance
        feedback = schemas.PreferenceFeedbackCreate(
            prompt="Test",
            response_chosen="A",
            response_rejected="B",
        )
        print(f"   - Created PreferenceFeedbackCreate: {feedback.feedback_type.value}")
        
        return True
    except Exception as e:
        print(f"❌ Schemas test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_feedback_direct():
    """Test feedback API module directly."""
    print("\n[TEST] Feedback API Module (Direct)")
    try:
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "feedback_api",
            "/Users/disanssebowabasalidde/Documents/GitHub/namel3ss-programming-language/namel3ss/ml/rlhf/api/feedback.py"
        )
        feedback_api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feedback_api)
        
        assert hasattr(feedback_api, 'router')
        assert hasattr(feedback_api, 'create_feedback')
        assert hasattr(feedback_api, 'list_feedback')
        assert hasattr(feedback_api, 'get_feedback')
        assert hasattr(feedback_api, 'delete_feedback')
        
        print("✅ Feedback API module loaded successfully")
        print(f"   - router: {feedback_api.router.prefix}")
        print(f"   - Endpoints: create, list, get, delete")
        return True
    except Exception as e:
        print(f"❌ Feedback API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_counts():
    """Test file counts and structure."""
    print("\n[TEST] File Structure")
    try:
        import os
        import glob
        
        base = "/Users/disanssebowabasalidde/Documents/GitHub/namel3ss-programming-language/namel3ss/ml/rlhf"
        
        # Count Python files
        py_files = glob.glob(f"{base}/**/*.py", recursive=True)
        py_files = [f for f in py_files if '__pycache__' not in f]
        
        # Count API files
        api_files = glob.glob(f"{base}/api/*.py")
        api_files = [f for f in api_files if '__pycache__' not in f and '__init__' not in f]
        
        print(f"✅ File structure validated")
        print(f"   - Total Python files: {len(py_files)}")
        print(f"   - API endpoint files: {len(api_files)}")
        print(f"   - Key files: models.py, database.py, schemas.py, exporters.py")
        print(f"   - API files: feedback.py, tasks.py, datasets.py, jobs.py")
        
        # Check key files exist
        assert os.path.exists(f"{base}/models.py")
        assert os.path.exists(f"{base}/database.py")
        assert os.path.exists(f"{base}/schemas.py")
        assert os.path.exists(f"{base}/exporters.py")
        assert os.path.exists(f"{base}/api/feedback.py")
        assert os.path.exists(f"{base}/api/tasks.py")
        assert os.path.exists(f"{base}/api/datasets.py")
        assert os.path.exists(f"{base}/api/jobs.py")
        
        return True
    except Exception as e:
        print(f"❌ File structure test failed: {e}")
        return False


def test_line_counts():
    """Test line counts."""
    print("\n[TEST] Line Counts")
    try:
        import os
        
        base = "/Users/disanssebowabasalidde/Documents/GitHub/namel3ss-programming-language/namel3ss/ml/rlhf"
        
        files = {
            "models.py": f"{base}/models.py",
            "database.py": f"{base}/database.py",
            "schemas.py": f"{base}/schemas.py",
            "exporters.py": f"{base}/exporters.py",
            "api/feedback.py": f"{base}/api/feedback.py",
            "api/tasks.py": f"{base}/api/tasks.py",
            "api/datasets.py": f"{base}/api/datasets.py",
            "api/jobs.py": f"{base}/api/jobs.py",
        }
        
        total = 0
        for name, path in files.items():
            with open(path, 'r') as f:
                lines = len(f.readlines())
                print(f"   - {name:25} {lines:4} lines")
                total += lines
        
        print(f"   {'='*25} {'='*4}")
        print(f"   {'Total API files:':25} {total:4} lines")
        
        print(f"✅ Line counts completed")
        return True
    except Exception as e:
        print(f"❌ Line counts failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Running All Tests")
    print("=" * 80)
    
    tests = [
        test_models_direct,
        test_database_direct,
        test_schemas_direct,
        test_api_feedback_direct,
        test_file_counts,
        test_line_counts,
    ]
    
    results = []
    for test_func in tests:
        try:
            passed = test_func()
            results.append(passed)
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_func.__name__}: {e}")
            results.append(False)
    
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
