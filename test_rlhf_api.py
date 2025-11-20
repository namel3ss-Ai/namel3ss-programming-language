"""
Integration tests for RLHF API endpoints.

Tests the /train_policy endpoint with real RLHF training.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from n3_server.main import app
from n3_server.db.models import Base, Project, Feedback, PolicyVersion
from n3_server.api.policies import get_db


# In-memory database for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def test_db():
    """Create test database."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session() as session:
        yield session
    
    await engine.dispose()


@pytest.fixture
async def test_project(test_db: AsyncSession):
    """Create test project."""
    project = Project(
        id="test_project",
        name="Test Project",
    )
    test_db.add(project)
    await test_db.commit()
    return project


@pytest.fixture
async def sample_feedback(test_db: AsyncSession, test_project: Project):
    """Create sample feedback data."""
    feedbacks = [
        Feedback(
            project_id=test_project.id,
            agent_id="test_agent",
            run_id=f"run_{i}",
            prompt=f"Question {i}",
            response=f"Answer {i}",
            score=0.5 + (i % 5) * 0.1,  # Varying scores
            notes=f"Notes for feedback {i}",
        )
        for i in range(15)  # More than minimum 10
    ]
    
    for f in feedbacks:
        test_db.add(f)
    
    await test_db.commit()
    return feedbacks


@pytest.fixture
def override_get_db(test_db: AsyncSession):
    """Override database dependency."""
    async def _get_db():
        yield test_db
    
    app.dependency_overrides[get_db] = _get_db
    yield
    app.dependency_overrides.clear()


class TestFeedbackEndpoint:
    """Test feedback submission endpoint."""
    
    @pytest.mark.asyncio
    async def test_submit_feedback(
        self,
        test_db: AsyncSession,
        test_project: Project,
        override_get_db,
    ):
        """Test submitting feedback."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/policies/feedback/test_agent",
                json={
                    "prompt": "What is 2+2?",
                    "response": "4",
                    "score": 1.0,
                    "runId": "run_123",
                    "notes": "Perfect answer",
                },
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["agentId"] == "test_agent"
        assert data["score"] == 1.0
    
    @pytest.mark.asyncio
    async def test_submit_feedback_invalid_score(
        self,
        test_db: AsyncSession,
        test_project: Project,
        override_get_db,
    ):
        """Test submitting feedback with invalid score."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Score > 1.0
            response = await client.post(
                "/api/policies/feedback/test_agent",
                json={
                    "prompt": "Test",
                    "response": "Test response",
                    "score": 1.5,
                    "runId": "run_123",
                },
            )
        
        assert response.status_code == 422  # Validation error


class TestPolicyListEndpoint:
    """Test policy listing endpoint."""
    
    @pytest.mark.asyncio
    async def test_list_policies_empty(
        self,
        test_db: AsyncSession,
        override_get_db,
    ):
        """Test listing policies when none exist."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/policies/test_agent")
        
        assert response.status_code == 200
        data = response.json()
        assert data == []
    
    @pytest.mark.asyncio
    async def test_list_policies_with_data(
        self,
        test_db: AsyncSession,
        sample_feedback,
        override_get_db,
    ):
        """Test listing policies after creating some."""
        # Create a policy
        policy = PolicyVersion(
            id="policy_1",
            agent_id="test_agent",
            version="v1",
            model_path="models/test_agent/v1",
            feedback_count=10,
            reward_mean=0.75,
            reward_std=0.15,
        )
        test_db.add(policy)
        await test_db.commit()
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/policies/test_agent")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["agentId"] == "test_agent"
        assert data[0]["version"] == "v1"
        assert data[0]["metrics"]["rewardMean"] == 0.75


class TestTrainPolicyEndpoint:
    """Test policy training endpoint."""
    
    @pytest.mark.asyncio
    async def test_train_policy_insufficient_feedback(
        self,
        test_db: AsyncSession,
        test_project: Project,
        override_get_db,
    ):
        """Test training with insufficient feedback."""
        # Add only 5 feedback items (less than minimum 10)
        feedbacks = [
            Feedback(
                project_id=test_project.id,
                agent_id="test_agent",
                run_id=f"run_{i}",
                prompt=f"Question {i}",
                response=f"Answer {i}",
                score=0.8,
            )
            for i in range(5)
        ]
        
        for f in feedbacks:
            test_db.add(f)
        await test_db.commit()
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/policies/train_policy/test_agent",
                json={
                    "dryRun": False,
                    "maxSteps": 10,
                    "learningRate": 1e-5,
                },
            )
        
        assert response.status_code == 400
        assert "Insufficient feedback" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_train_policy_dry_run(
        self,
        test_db: AsyncSession,
        sample_feedback,
        override_get_db,
    ):
        """Test dry run (estimation) mode."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/policies/train_policy/test_agent",
                json={
                    "dryRun": True,
                    "maxSteps": 100,
                    "learningRate": 1e-5,
                },
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "dry_run"
        assert data["feedbackCount"] == 15
        assert data["estimatedSteps"] == 100
        assert "estimatedTimeMinutes" in data
        assert "scoreMean" in data
        assert "scoreStd" in data
        assert "modelConfig" in data
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_train_policy_actual_training(
        self,
        test_db: AsyncSession,
        sample_feedback,
        override_get_db,
        tmp_path,
        monkeypatch,
    ):
        """Test actual training (very short for testing)."""
        # Override models directory to use tmp_path
        monkeypatch.setenv("MODELS_DIR", str(tmp_path))
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/policies/train_policy/test_agent",
                json={
                    "dryRun": False,
                    "maxSteps": 5,  # Very short for testing
                    "learningRate": 1e-5,
                },
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "trained"
        assert "policyId" in data
        assert data["version"].startswith("v")
        assert "modelPath" in data
        assert "rewardModelPath" in data
        assert data["feedbackCount"] == 15
        assert data["trainingSteps"] > 0
        assert "rewardMean" in data
        assert "rewardStd" in data
        
        # Verify policy was saved to database
        from sqlalchemy import select
        result = await test_db.execute(
            select(PolicyVersion).where(
                PolicyVersion.id == data["policyId"]
            )
        )
        policy = result.scalar_one_or_none()
        
        assert policy is not None
        assert policy.agent_id == "test_agent"
        assert policy.version == data["version"]


class TestEndToEndWorkflow:
    """Test complete RLHF workflow."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_complete_rlhf_workflow(
        self,
        test_db: AsyncSession,
        test_project: Project,
        override_get_db,
        tmp_path,
        monkeypatch,
    ):
        """Test complete workflow: feedback → training → listing."""
        monkeypatch.setenv("MODELS_DIR", str(tmp_path))
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Step 1: Submit feedback
            for i in range(12):
                response = await client.post(
                    "/api/policies/feedback/workflow_agent",
                    json={
                        "prompt": f"Question {i}",
                        "response": f"Answer {i}",
                        "score": 0.5 + (i % 5) * 0.1,
                        "runId": f"run_{i}",
                        "notes": f"Feedback {i}",
                    },
                )
                assert response.status_code == 200
            
            # Step 2: Dry run to estimate
            response = await client.post(
                "/api/policies/train_policy/workflow_agent",
                json={
                    "dryRun": True,
                    "maxSteps": 5,
                    "learningRate": 1e-5,
                },
            )
            assert response.status_code == 200
            estimate = response.json()
            assert estimate["feedbackCount"] == 12
            
            # Step 3: Train policy
            response = await client.post(
                "/api/policies/train_policy/workflow_agent",
                json={
                    "dryRun": False,
                    "maxSteps": 5,
                    "learningRate": 1e-5,
                },
            )
            assert response.status_code == 200
            training_result = response.json()
            assert training_result["status"] == "trained"
            policy_id = training_result["policyId"]
            
            # Step 4: List policies
            response = await client.get("/api/policies/workflow_agent")
            assert response.status_code == 200
            policies = response.json()
            assert len(policies) == 1
            assert policies[0]["id"] == policy_id
            assert policies[0]["agentId"] == "workflow_agent"
            assert policies[0]["metrics"]["feedbackCount"] == 12
            
            # Step 5: Submit more feedback and train again
            for i in range(12, 20):
                response = await client.post(
                    "/api/policies/feedback/workflow_agent",
                    json={
                        "prompt": f"Question {i}",
                        "response": f"Answer {i}",
                        "score": 0.6 + (i % 4) * 0.1,
                        "runId": f"run_{i}",
                    },
                )
                assert response.status_code == 200
            
            # Train second version
            response = await client.post(
                "/api/policies/train_policy/workflow_agent",
                json={
                    "dryRun": False,
                    "maxSteps": 5,
                    "learningRate": 1e-5,
                },
            )
            assert response.status_code == 200
            second_training = response.json()
            assert second_training["feedbackCount"] == 20
            
            # List both policies
            response = await client.get("/api/policies/workflow_agent")
            assert response.status_code == 200
            policies = response.json()
            assert len(policies) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
