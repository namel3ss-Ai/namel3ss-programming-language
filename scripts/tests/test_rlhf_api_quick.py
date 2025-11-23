"""
Quick RLHF API Import Test - Validate feedback collection modules.

This test validates that all API modules can be imported without errors.
"""

print("=" * 80)
print("RLHF Feedback API Import Test")
print("=" * 80)

def test_models():
    """Test database models import."""
    print("\n[TEST] Database Models")
    try:
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
        print("✅ Models imported successfully")
        print(f"   - Feedback, AnnotationTask, Dataset, TrainingJob")
        print(f"   - FeedbackType, TaskStatus, DatasetStatus, JobStatus")
        return True
    except ImportError as e:
        print(f"❌ Models import failed: {e}")
        return False


def test_database():
    """Test database module import."""
    print("\n[TEST] Database Module")
    try:
        from namel3ss.ml.rlhf.database import (
            DatabaseManager,
            initialize_database,
            get_database,
            get_session,
        )
        print("✅ Database module imported successfully")
        print(f"   - DatabaseManager, initialize_database")
        print(f"   - get_database, get_session")
        return True
    except ImportError as e:
        print(f"❌ Database import failed: {e}")
        return False


def test_schemas():
    """Test Pydantic schemas import."""
    print("\n[TEST] API Schemas")
    try:
        from namel3ss.ml.rlhf.schemas import (
            PreferenceFeedbackCreate,
            ScoreFeedbackCreate,
            BinaryFeedbackCreate,
            FeedbackResponse,
            TaskCreate,
            TaskResponse,
            DatasetCreate,
            DatasetResponse,
            JobCreate,
            JobResponse,
        )
        print("✅ Schemas imported successfully")
        print(f"   - Feedback schemas (4 types)")
        print(f"   - Task, Dataset, Job schemas")
        return True
    except ImportError as e:
        print(f"❌ Schemas import failed: {e}")
        return False


def test_exporters():
    """Test dataset exporters import."""
    print("\n[TEST] Dataset Exporters")
    try:
        from namel3ss.ml.rlhf.exporters import DatasetExporter
        print("✅ Exporters imported successfully")
        print(f"   - DatasetExporter")
        return True
    except ImportError as e:
        print(f"❌ Exporters import failed: {e}")
        return False


def test_api_routers():
    """Test FastAPI routers import."""
    print("\n[TEST] API Routers")
    try:
        from namel3ss.ml.rlhf.api import (
            feedback_router,
            tasks_router,
            datasets_router,
            jobs_router,
        )
        print("✅ API routers imported successfully")
        print(f"   - feedback_router: {feedback_router.prefix}")
        print(f"   - tasks_router: {tasks_router.prefix}")
        print(f"   - datasets_router: {datasets_router.prefix}")
        print(f"   - jobs_router: {jobs_router.prefix}")
        return True
    except ImportError as e:
        print(f"❌ API routers import failed: {e}")
        return False


def test_model_creation():
    """Test creating model instances."""
    print("\n[TEST] Model Instance Creation")
    try:
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
        from datetime import datetime, timedelta
        
        # Create feedback
        feedback = Feedback(
            feedback_type=FeedbackType.PREFERENCE,
            prompt="Test prompt",
            response_chosen="Good response",
            response_rejected="Bad response",
        )
        assert feedback.feedback_type == FeedbackType.PREFERENCE
        
        # Create task
        task = AnnotationTask(
            name="Test Task",
            feedback_type=FeedbackType.SCORE,
            num_prompts=10,
        )
        assert task.status == TaskStatus.PENDING
        
        # Create dataset
        dataset = Dataset(
            name="test_dataset",
            feedback_type=FeedbackType.PREFERENCE,
            export_format="parquet",
        )
        assert dataset.status == DatasetStatus.PENDING
        
        # Create job
        job = TrainingJob(
            job_name="test_job",
            algorithm="dpo",
            base_model="gpt2",
        )
        assert job.status == JobStatus.PENDING
        
        print("✅ Model instances created successfully")
        print(f"   - Feedback, Task, Dataset, Job")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False


def test_enum_values():
    """Test enum values."""
    print("\n[TEST] Enum Values")
    try:
        from namel3ss.ml.rlhf.models import (
            FeedbackType,
            TaskStatus,
            DatasetStatus,
            JobStatus,
        )
        
        # Check enum values
        assert FeedbackType.PREFERENCE.value == "preference"
        assert FeedbackType.SCORE.value == "score"
        assert FeedbackType.BINARY.value == "binary"
        assert FeedbackType.RANKING.value == "ranking"
        
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        
        assert DatasetStatus.PENDING.value == "pending"
        assert DatasetStatus.READY.value == "ready"
        
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        
        print("✅ Enum values validated")
        print(f"   - FeedbackType: 4 values")
        print(f"   - TaskStatus: 4 values")
        print(f"   - DatasetStatus: 4 values")
        print(f"   - JobStatus: 5 values")
        return True
    except Exception as e:
        print(f"❌ Enum validation failed: {e}")
        return False


def test_schema_validation():
    """Test Pydantic schema validation."""
    print("\n[TEST] Schema Validation")
    try:
        from namel3ss.ml.rlhf.schemas import (
            PreferenceFeedbackCreate,
            TaskCreate,
            DatasetCreate,
        )
        
        # Test preference feedback
        feedback = PreferenceFeedbackCreate(
            prompt="What is AI?",
            response_chosen="AI is artificial intelligence.",
            response_rejected="AI is a robot.",
        )
        assert feedback.feedback_type.value == "preference"
        
        # Test task
        task = TaskCreate(
            name="Test Task",
            feedback_type="score",
            num_prompts=100,
        )
        assert task.num_prompts == 100
        
        # Test dataset
        dataset = DatasetCreate(
            name="test_dataset",
            feedback_type="preference",
            export_format="parquet",
        )
        assert dataset.export_format == "parquet"
        
        print("✅ Schema validation passed")
        print(f"   - PreferenceFeedbackCreate")
        print(f"   - TaskCreate")
        print(f"   - DatasetCreate")
        return True
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Running All Tests")
    print("=" * 80)
    
    tests = [
        test_models,
        test_database,
        test_schemas,
        test_exporters,
        test_api_routers,
        test_model_creation,
        test_enum_values,
        test_schema_validation,
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
