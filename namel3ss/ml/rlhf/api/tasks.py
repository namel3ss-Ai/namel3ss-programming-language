"""
RLHF Tasks API - Endpoints for managing annotation tasks.

Provides:
- POST /api/rlhf/tasks - Create task
- GET /api/rlhf/tasks - List tasks
- GET /api/rlhf/tasks/{id} - Get task
- PATCH /api/rlhf/tasks/{id} - Update task
- POST /api/rlhf/tasks/{id}/start - Start task
- POST /api/rlhf/tasks/{id}/complete - Complete task
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models import AnnotationTask, TaskStatus, FeedbackType
from ..schemas import (
    TaskCreate,
    TaskUpdate,
    TaskResponse,
    TaskListResponse,
    TaskFilters,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rlhf/tasks", tags=["tasks"])


@router.post(
    "",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create task",
    description="Create new annotation task"
)
async def create_task(
    task: TaskCreate,
    session: AsyncSession = Depends(get_session),
):
    """
    Create annotation task.
    
    Args:
        task: Task configuration
    
    Returns:
        Created task with ID
    """
    try:
        feedback_type = FeedbackType(task.feedback_type.value)
        
        db_task = AnnotationTask(
            name=task.name,
            description=task.description,
            feedback_type=feedback_type,
            instructions=task.instructions,
            num_prompts=task.num_prompts,
            assigned_to=task.assigned_to,
            deadline=task.deadline,
            model_id=task.model_id,
            metadata=task.metadata,
        )
        
        session.add(db_task)
        await session.commit()
        await session.refresh(db_task)
        
        logger.info(f"Created task {db_task.id}: {db_task.name}")
        
        return TaskResponse.model_validate(db_task)
        
    except Exception as e:
        await session.rollback()
        logger.error(f"Failed to create task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create task: {str(e)}"
        )


@router.get(
    "",
    response_model=TaskListResponse,
    summary="List tasks",
    description="List tasks with filtering and pagination"
)
async def list_tasks(
    filters: TaskFilters = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """
    List tasks with filters.
    
    Returns:
        Paginated list of tasks
    """
    try:
        query = select(AnnotationTask)
        conditions = []
        
        if filters.status:
            conditions.append(AnnotationTask.status == TaskStatus(filters.status.value))
        if filters.assigned_to:
            conditions.append(AnnotationTask.assigned_to == filters.assigned_to)
        if filters.feedback_type:
            conditions.append(AnnotationTask.feedback_type == FeedbackType(filters.feedback_type.value))
        if filters.created_after:
            conditions.append(AnnotationTask.created_at >= filters.created_after)
        if filters.created_before:
            conditions.append(AnnotationTask.created_at <= filters.created_before)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Count total
        count_query = select(func.count()).select_from(AnnotationTask)
        if conditions:
            count_query = count_query.where(and_(*conditions))
        total_result = await session.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        offset = (filters.page - 1) * filters.page_size
        query = query.order_by(AnnotationTask.created_at.desc())
        query = query.offset(offset).limit(filters.page_size)
        
        result = await session.execute(query)
        tasks = result.scalars().all()
        
        has_next = offset + len(tasks) < total
        
        return TaskListResponse(
            items=[TaskResponse.model_validate(t) for t in tasks],
            total=total,
            page=filters.page,
            page_size=filters.page_size,
            has_next=has_next,
        )
        
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tasks: {str(e)}"
        )


@router.get(
    "/{task_id}",
    response_model=TaskResponse,
    summary="Get task",
    description="Get specific task by ID"
)
async def get_task(
    task_id: int,
    session: AsyncSession = Depends(get_session),
):
    """
    Get task by ID.
    
    Args:
        task_id: Task ID
    
    Returns:
        Task details
    """
    try:
        result = await session.execute(
            select(AnnotationTask).where(AnnotationTask.id == task_id)
        )
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        return TaskResponse.model_validate(task)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task: {str(e)}"
        )


@router.patch(
    "/{task_id}",
    response_model=TaskResponse,
    summary="Update task",
    description="Update task fields"
)
async def update_task(
    task_id: int,
    update: TaskUpdate,
    session: AsyncSession = Depends(get_session),
):
    """
    Update task.
    
    Args:
        task_id: Task ID
        update: Fields to update
    
    Returns:
        Updated task
    """
    try:
        result = await session.execute(
            select(AnnotationTask).where(AnnotationTask.id == task_id)
        )
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        # Update fields
        if update.status:
            task.status = TaskStatus(update.status.value)
        if update.assigned_to is not None:
            task.assigned_to = update.assigned_to
        if update.deadline is not None:
            task.deadline = update.deadline
        if update.started_at is not None:
            task.started_at = update.started_at
        if update.completed_at is not None:
            task.completed_at = update.completed_at
        
        await session.commit()
        await session.refresh(task)
        
        logger.info(f"Updated task {task_id}")
        
        return TaskResponse.model_validate(task)
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Failed to update task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update task: {str(e)}"
        )


@router.post(
    "/{task_id}/start",
    response_model=TaskResponse,
    summary="Start task",
    description="Mark task as in progress"
)
async def start_task(
    task_id: int,
    session: AsyncSession = Depends(get_session),
):
    """
    Start task (mark as in progress).
    
    Args:
        task_id: Task ID
    
    Returns:
        Updated task
    """
    try:
        result = await session.execute(
            select(AnnotationTask).where(AnnotationTask.id == task_id)
        )
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        if task.status != TaskStatus.PENDING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Task must be PENDING to start (current: {task.status.value})"
            )
        
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()
        
        await session.commit()
        await session.refresh(task)
        
        logger.info(f"Started task {task_id}")
        
        return TaskResponse.model_validate(task)
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Failed to start task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start task: {str(e)}"
        )


@router.post(
    "/{task_id}/complete",
    response_model=TaskResponse,
    summary="Complete task",
    description="Mark task as completed"
)
async def complete_task(
    task_id: int,
    session: AsyncSession = Depends(get_session),
):
    """
    Complete task.
    
    Args:
        task_id: Task ID
    
    Returns:
        Updated task
    """
    try:
        result = await session.execute(
            select(AnnotationTask).where(AnnotationTask.id == task_id)
        )
        task = result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        if task.status not in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Task already completed or expired"
            )
        
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.utcnow()
        
        await session.commit()
        await session.refresh(task)
        
        logger.info(f"Completed task {task_id}")
        
        return TaskResponse.model_validate(task)
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Failed to complete task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete task: {str(e)}"
        )
