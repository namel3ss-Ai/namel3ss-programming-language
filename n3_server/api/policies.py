from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from nanoid import generate

from n3_server.db.session import get_db
from n3_server.db.models import Feedback, PolicyVersion

router = APIRouter()


class FeedbackSubmission(BaseModel):
    prompt: str
    response: str
    score: float
    notes: str | None = None
    runId: str


class PolicyMetadata(BaseModel):
    id: str
    agentId: str
    version: str
    createdAt: datetime
    trainedOn: dict[str, float] | None = None


class PolicyTrainingRequest(BaseModel):
    dryRun: bool = False
    maxSteps: int = 1000
    learningRate: float = 1e-5


@router.post("/feedback/{agent_id}")
async def submit_feedback(
    agent_id: str,
    feedback: FeedbackSubmission,
    db: AsyncSession = Depends(get_db),
):
    """Submit feedback for an agent response."""
    feedback_entry = Feedback(
        id=generate(size=12),
        project_id="demo",  # TODO: Get from auth context
        agent_id=agent_id,
        run_id=feedback.runId,
        prompt=feedback.prompt,
        response=feedback.response,
        score=feedback.score,
        notes=feedback.notes,
    )
    
    db.add(feedback_entry)
    await db.commit()
    
    return {"status": "submitted", "id": feedback_entry.id}


@router.get("/policies/{agent_id}", response_model=list[PolicyMetadata])
async def list_policies(agent_id: str, db: AsyncSession = Depends(get_db)):
    """List trained policy versions for an agent."""
    result = await db.execute(
        select(PolicyVersion)
        .where(PolicyVersion.agent_id == agent_id)
        .order_by(PolicyVersion.created_at.desc())
    )
    policies = result.scalars().all()
    
    return [
        PolicyMetadata(
            id=p.id,
            agentId=p.agent_id,
            version=p.version,
            createdAt=p.created_at,
            trainedOn={
                "feedbackCount": p.feedback_count,
                "rewardMean": p.reward_mean or 0.0,
                "rewardStd": p.reward_std or 0.0,
            } if p.feedback_count > 0 else None,
        )
        for p in policies
    ]


@router.post("/train_policy/{agent_id}")
async def train_policy(
    agent_id: str,
    request: PolicyTrainingRequest,
    db: AsyncSession = Depends(get_db),
):
    """Train a new policy using RLHF/PPO with TRL library."""
    from pathlib import Path
    from n3_server.rlhf import RLHFTrainer, TrainingConfig
    
    # Get feedback for agent
    result = await db.execute(
        select(Feedback).where(Feedback.agent_id == agent_id)
    )
    feedbacks_orm = result.scalars().all()
    
    if len(feedbacks_orm) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient feedback: {len(feedbacks_orm)} samples (minimum 10 required)",
        )
    
    # Convert ORM objects to dictionaries
    feedbacks = [
        {
            "prompt": f.prompt,
            "response": f.response,
            "score": f.score,
            "run_id": f.run_id,
            "notes": f.notes,
        }
        for f in feedbacks_orm
    ]
    
    # Create training configuration
    config = TrainingConfig(
        base_model="gpt2",  # Default model, can be parameterized
        use_lora=True,
        learning_rate=request.learningRate,
        max_steps=request.maxSteps,
        batch_size=8,
        train_reward_model_first=True,
    )
    
    # Initialize trainer
    trainer = RLHFTrainer(config)
    
    # Handle dry run
    if request.dryRun:
        estimate = trainer.estimate_training(feedbacks)
        return {
            "status": "dry_run",
            "feedbackCount": len(feedbacks),
            "estimatedSteps": estimate["estimated_steps"],
            "estimatedTimeMinutes": estimate["estimated_time_minutes"],
            "scoreMean": estimate["score_mean"],
            "scoreStd": estimate["score_std"],
            "modelConfig": estimate["model_config"],
        }
    
    # Train policy with RLHF
    policy_id = generate(size=12)
    version = f"v{len(feedbacks)}"
    output_dir = Path("models") / agent_id / version
    
    try:
        # Run training
        training_result = trainer.train(
            feedbacks=feedbacks,
            output_dir=output_dir,
        )
        
        # Create policy version record
        policy = PolicyVersion(
            id=policy_id,
            agent_id=agent_id,
            version=version,
            model_path=training_result.model_path,
            feedback_count=training_result.feedback_count,
            reward_mean=training_result.final_reward_mean,
            reward_std=training_result.final_reward_std,
        )
        
        db.add(policy)
        await db.commit()
        
        return {
            "status": "trained",
            "policyId": policy_id,
            "version": version,
            "modelPath": training_result.model_path,
            "rewardModelPath": training_result.reward_model_path,
            "feedbackCount": training_result.feedback_count,
            "trainingSteps": training_result.training_steps,
            "rewardMean": training_result.final_reward_mean,
            "rewardStd": training_result.final_reward_std,
        }
    
    except Exception as e:
        # If training fails, return error
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}",
        )
