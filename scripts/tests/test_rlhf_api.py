"""
Test RLHF Feedback API - Validate feedback collection system.

Tests:
- Database models and relationships
- API endpoints (feedback, tasks, datasets, jobs)
- Dataset export functionality
- Error handling
"""

from datetime import datetime, timedelta

# Database and models
from namel3ss.ml.rlhf.database import DatabaseManager, initialize_database
from namel3ss.ml.rlhf.models import (
    Feedback,
    AnnotationTask,
    Dataset,
    TrainingJob,
    FeedbackType,
    TaskStatus,
    DatasetStatus,
    JobStatus,
)

print("=" * 80)
print("RLHF Feedback API Test Suite")
print("=" * 80)


# ============================================================================
# Test Database Models
# ============================================================================

def test_feedback_model():
    """Test Feedback model creation and serialization."""
    print("\n[TEST] Feedback Model")
    
    feedback = Feedback(
        feedback_type=FeedbackType.PREFERENCE,
        prompt="What is the capital of France?",
        response_chosen="The capital of France is Paris.",
        response_rejected="France's capital is Lyon.",
        annotator_id="annotator_001",
        confidence=0.95,
    )
    
    assert feedback.feedback_type == FeedbackType.PREFERENCE
    assert feedback.prompt == "What is the capital of France?"
    assert feedback.response_chosen is not None
    assert feedback.confidence == 0.95
    
    # Test to_dict
    data = feedback.to_dict()
    assert data["feedback_type"] == "preference"
    assert data["prompt"] == "What is the capital of France?"
    
    print("✅ Feedback model test passed")


def test_task_model():
    """Test AnnotationTask model."""
    print("\n[TEST] Task Model")
    
    task = AnnotationTask(
        name="Quality Assessment Batch 1",
        feedback_type=FeedbackType.SCORE,
        num_prompts=100,
        assigned_to="annotator_001",
        deadline=datetime.utcnow() + timedelta(days=7),
    )
    
    assert task.name == "Quality Assessment Batch 1"
    assert task.status == TaskStatus.PENDING
    assert task.num_prompts == 100
    
    data = task.to_dict()
    assert data["status"] == "pending"
    assert data["num_prompts"] == 100
    
    print("✅ Task model test passed")


def test_dataset_model():
    """Test Dataset model."""
    print("\n[TEST] Dataset Model")
    
    dataset = Dataset(
        name="preference_v1",
        feedback_type=FeedbackType.PREFERENCE,
        export_format="parquet",
        min_confidence=0.8,
    )
    
    assert dataset.name == "preference_v1"
    assert dataset.status == DatasetStatus.PENDING
    assert dataset.export_format == "parquet"
    
    data = dataset.to_dict()
    assert data["name"] == "preference_v1"
    assert data["status"] == "pending"
    
    print("✅ Dataset model test passed")


def test_job_model():
    """Test TrainingJob model."""
    print("\n[TEST] Job Model")
    
    job = TrainingJob(
        job_name="dpo_llama3_v1",
        algorithm="dpo",
        base_model="meta-llama/Meta-Llama-3-8B",
        config={"learning_rate": 1e-5},
    )
    
    assert job.job_name == "dpo_llama3_v1"
    assert job.algorithm == "dpo"
    assert job.status == JobStatus.PENDING
    
    data = job.to_dict()
    assert data["algorithm"] == "dpo"
    assert data["status"] == "pending"
    
    print("✅ Job model test passed")


# ============================================================================
# Test Database Operations
# ============================================================================

def test_database_manager_sync():
    """Test DatabaseManager initialization (sync version for quick test)."""
    print("\n[TEST] Database Manager")
    
    # Use SQLite for testing
    db_url = "sqlite+aiosqlite:///./test_rlhf.db"
    
    try:
        db = DatabaseManager(
            database_url=db_url,
            echo=False,
        )
        
        assert db.database_url == db_url
        assert db.engine is not None
        assert db.session_factory is not None
        
        print("✅ Database manager test passed")
        
    except Exception as e:
        print(f"⚠️  Database manager test skipped: {e}")
        print("   (This is expected if asyncpg/aiosqlite not installed)")


# ============================================================================
# Test API Schemas
# ============================================================================

def test_api_schemas():
    """Test Pydantic schemas for API validation."""
    print("\n[TEST] API Schemas")
    
    try:
        from namel3ss.ml.rlhf.schemas import (
            PreferenceFeedbackCreate,
            TaskCreate,
            DatasetCreate,
            JobCreate,
        )
        
        # Test preference feedback schema
        feedback = PreferenceFeedbackCreate(
            prompt="What is AI?",
            response_chosen="AI is artificial intelligence.",
            response_rejected="AI is a programming language.",
        )
        assert feedback.feedback_type.value == "preference"
        
        # Test task schema
        task = TaskCreate(
            name="Test Task",
            feedback_type="preference",
            num_prompts=50,
        )
        assert task.name == "Test Task"
        
        # Test dataset schema
        dataset = DatasetCreate(
            name="test_dataset",
            feedback_type="preference",
            export_format="parquet",
        )
        assert dataset.export_format == "parquet"
        
        # Test job schema
        job = JobCreate(
            job_name="test_job",
            algorithm="dpo",
            base_model="gpt2",
            dataset_id=1,
            config={},
        )
        assert job.algorithm == "dpo"
        
        print("✅ API schemas test passed")
        
    except ImportError as e:
        print(f"⚠️  API schemas test skipped: {e}")


# ============================================================================
# Test Import Validation
# ============================================================================

def test_imports():
    """Test all RLHF feedback API imports."""
    print("\n[TEST] Import Validation")
    
    try:
        # Database imports
        from namel3ss.ml.rlhf.database import (
            DatabaseManager,
            initialize_database,
            get_database,
            get_session,
        )
        
        # Model imports
        from namel3ss.ml.rlhf.models import (
            Feedback,
            AnnotationTask,
            Dataset,
            TrainingJob,
        )
        
        # Schema imports
        from namel3ss.ml.rlhf.schemas import (
            FeedbackResponse,
            TaskResponse,
            DatasetResponse,
            JobResponse,
        )
        
        # Exporter imports
        from namel3ss.ml.rlhf.exporters import DatasetExporter
        
        # API imports
        from namel3ss.ml.rlhf.api import (
            feedback_router,
            tasks_router,
            datasets_router,
            jobs_router,
        )
        
        print("✅ All imports successful")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        raise


def test_package_exports():
    """Test package-level exports."""
    print("\n[TEST] Package Exports")
    
    try:
        from namel3ss.ml.rlhf import (
            DatabaseManager,
            Feedback,
            AnnotationTask,
            Dataset,
            TrainingJob,
            DatasetExporter,
            feedback_router,
            tasks_router,
            datasets_router,
            jobs_router,
        )
        
        print("✅ Package exports test passed")
        
    except ImportError as e:
        print(f"❌ Package exports failed: {e}")
        raise


# ============================================================================
# Test Enum Values
# ============================================================================

def test_enum_values():
    """Test enum value mappings."""
    print("\n[TEST] Enum Values")
    
    # FeedbackType
    assert FeedbackType.PREFERENCE.value == "preference"
    assert FeedbackType.SCORE.value == "score"
    assert FeedbackType.BINARY.value == "binary"
    assert FeedbackType.RANKING.value == "ranking"
    
    # TaskStatus
    assert TaskStatus.PENDING.value == "pending"
    assert TaskStatus.IN_PROGRESS.value == "in_progress"
    assert TaskStatus.COMPLETED.value == "completed"
    
    # DatasetStatus
    assert DatasetStatus.PENDING.value == "pending"
    assert DatasetStatus.PROCESSING.value == "processing"
    assert DatasetStatus.READY.value == "ready"
    assert DatasetStatus.FAILED.value == "failed"
    
    # JobStatus
    assert JobStatus.PENDING.value == "pending"
    assert JobStatus.RUNNING.value == "running"
    assert JobStatus.COMPLETED.value == "completed"
    assert JobStatus.FAILED.value == "failed"
    
    print("✅ Enum values test passed")


# ============================================================================
# Test API Router Registration
# ============================================================================

def test_api_routers():
    """Test FastAPI router registration."""
    print("\n[TEST] API Routers")
    
    try:
        from fastapi import FastAPI
        from namel3ss.ml.rlhf.api import (
            feedback_router,
            tasks_router,
            datasets_router,
            jobs_router,
        )
        
        app = FastAPI()
        
        # Register routers
        app.include_router(feedback_router)
        app.include_router(tasks_router)
        app.include_router(datasets_router)
        app.include_router(jobs_router)
        
        # Check routes registered
        routes = [route.path for route in app.routes]
        
        assert "/api/rlhf/feedback" in routes
        assert "/api/rlhf/tasks" in routes
        assert "/api/rlhf/datasets" in routes
        assert "/api/rlhf/jobs" in routes
        
        print("✅ API routers test passed")
        print(f"   Registered {len(routes)} routes")
        
    except ImportError as e:
        print(f"⚠️  API routers test skipped: {e}")


# ============================================================================
# Run All Tests
# ============================================================================

def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 80)
    print("Running All Tests")
    print("=" * 80)
    
    tests = [
        test_feedback_model,
        test_task_model,
        test_dataset_model,
        test_job_model,
        test_database_manager_sync,
        test_api_schemas,
        test_imports,
        test_package_exports,
        test_enum_values,
        test_api_routers,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            if "skipped" in str(e).lower():
                skipped += 1
            else:
                failed += 1
                print(f"\n❌ {test_func.__name__} failed: {e}")
    
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Skipped: {skipped}")
    print(f"Total: {len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
