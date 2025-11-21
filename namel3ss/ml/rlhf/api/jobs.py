"""
RLHF Jobs API - Endpoints for managing training jobs.

Provides:
- POST /api/rlhf/jobs - Create job
- GET /api/rlhf/jobs - List jobs
- GET /api/rlhf/jobs/{id} - Get job
- PATCH /api/rlhf/jobs/{id} - Update job
- POST /api/rlhf/jobs/{id}/start - Start job
- POST /api/rlhf/jobs/{id}/stop - Stop job
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models import TrainingJob, JobStatus, Dataset
from ..schemas import (
    JobCreate,
    JobUpdate,
    JobResponse,
    JobListResponse,
    JobFilters,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rlhf/jobs", tags=["jobs"])


@router.post(
    "",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create job",
    description="Create RLHF training job"
)
async def create_job(
    job: JobCreate,
    session: AsyncSession = Depends(get_session),
):
    """
    Create training job.
    
    Args:
        job: Job configuration
    
    Returns:
        Created job
    """
    try:
        # Check if name already exists
        result = await session.execute(
            select(TrainingJob).where(TrainingJob.job_name == job.job_name)
        )
        existing = result.scalar_one_or_none()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job with name '{job.job_name}' already exists"
            )
        
        # Verify dataset exists
        dataset_result = await session.execute(
            select(Dataset).where(Dataset.id == job.dataset_id)
        )
        dataset = dataset_result.scalar_one_or_none()
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {job.dataset_id} not found"
            )
        
        db_job = TrainingJob(
            job_name=job.job_name,
            algorithm=job.algorithm,
            base_model=job.base_model,
            reward_model=job.reward_model,
            dataset_id=job.dataset_id,
            config=job.config,
            metadata=job.metadata,
        )
        
        session.add(db_job)
        await session.commit()
        await session.refresh(db_job)
        
        logger.info(f"Created job {db_job.id}: {db_job.job_name}")
        
        return JobResponse.model_validate(db_job)
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Failed to create job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {str(e)}"
        )


@router.get(
    "",
    response_model=JobListResponse,
    summary="List jobs",
    description="List jobs with filtering and pagination"
)
async def list_jobs(
    filters: JobFilters = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """
    List jobs with filters.
    
    Returns:
        Paginated list of jobs
    """
    try:
        query = select(TrainingJob)
        conditions = []
        
        if filters.status:
            conditions.append(TrainingJob.status == JobStatus(filters.status.value))
        if filters.algorithm:
            conditions.append(TrainingJob.algorithm == filters.algorithm)
        if filters.dataset_id:
            conditions.append(TrainingJob.dataset_id == filters.dataset_id)
        if filters.created_after:
            conditions.append(TrainingJob.created_at >= filters.created_after)
        if filters.created_before:
            conditions.append(TrainingJob.created_at <= filters.created_before)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Count total
        count_query = select(func.count()).select_from(TrainingJob)
        if conditions:
            count_query = count_query.where(and_(*conditions))
        total_result = await session.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        offset = (filters.page - 1) * filters.page_size
        query = query.order_by(TrainingJob.created_at.desc())
        query = query.offset(offset).limit(filters.page_size)
        
        result = await session.execute(query)
        jobs = result.scalars().all()
        
        has_next = offset + len(jobs) < total
        
        return JobListResponse(
            items=[JobResponse.model_validate(j) for j in jobs],
            total=total,
            page=filters.page,
            page_size=filters.page_size,
            has_next=has_next,
        )
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {str(e)}"
        )


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get job",
    description="Get specific job by ID"
)
async def get_job(
    job_id: int,
    session: AsyncSession = Depends(get_session),
):
    """
    Get job by ID.
    
    Args:
        job_id: Job ID
    
    Returns:
        Job details
    """
    try:
        result = await session.execute(
            select(TrainingJob).where(TrainingJob.id == job_id)
        )
        job = result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        return JobResponse.model_validate(job)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job: {str(e)}"
        )


@router.patch(
    "/{job_id}",
    response_model=JobResponse,
    summary="Update job",
    description="Update job fields"
)
async def update_job(
    job_id: int,
    update: JobUpdate,
    session: AsyncSession = Depends(get_session),
):
    """
    Update job.
    
    Args:
        job_id: Job ID
        update: Fields to update
    
    Returns:
        Updated job
    """
    try:
        result = await session.execute(
            select(TrainingJob).where(TrainingJob.id == job_id)
        )
        job = result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        # Update fields
        if update.status:
            job.status = JobStatus(update.status.value)
        if update.started_at is not None:
            job.started_at = update.started_at
        if update.completed_at is not None:
            job.completed_at = update.completed_at
        if update.final_loss is not None:
            job.final_loss = update.final_loss
        if update.total_steps is not None:
            job.total_steps = update.total_steps
        if update.duration_seconds is not None:
            job.duration_seconds = update.duration_seconds
        if update.checkpoint_path is not None:
            job.checkpoint_path = update.checkpoint_path
        if update.output_model is not None:
            job.output_model = update.output_model
        if update.metrics is not None:
            job.metrics = update.metrics
        if update.error_message is not None:
            job.error_message = update.error_message
        if update.error_type is not None:
            job.error_type = update.error_type
        
        await session.commit()
        await session.refresh(job)
        
        logger.info(f"Updated job {job_id}")
        
        return JobResponse.model_validate(job)
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Failed to update job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update job: {str(e)}"
        )
