"""
RLHF Feedback API - Endpoints for collecting and querying feedback.

Provides:
- POST /api/rlhf/feedback - Submit feedback
- GET /api/rlhf/feedback - List feedback with filters
- GET /api/rlhf/feedback/{id} - Get specific feedback
- DELETE /api/rlhf/feedback/{id} - Delete feedback
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models import Feedback, FeedbackType
from ..schemas import (
    PreferenceFeedbackCreate,
    ScoreFeedbackCreate,
    BinaryFeedbackCreate,
    RankingFeedbackCreate,
    FeedbackResponse,
    FeedbackListResponse,
    FeedbackFilters,
)
from ..errors import RLHFError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rlhf/feedback", tags=["feedback"])


@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit feedback",
    description="Submit human feedback on model responses"
)
async def create_feedback(
    feedback: PreferenceFeedbackCreate | ScoreFeedbackCreate | BinaryFeedbackCreate | RankingFeedbackCreate,
    session: AsyncSession = Depends(get_session),
):
    """
    Submit human feedback.
    
    Supports multiple feedback types:
    - preference: Pairwise preference (chosen vs rejected)
    - score: Numeric score
    - binary: Thumbs up/down
    - ranking: Ranking of multiple responses
    
    Returns:
        Created feedback with ID
    """
    try:
        # Map Pydantic schema to SQLAlchemy model
        feedback_type = FeedbackType(feedback.feedback_type.value)
        
        db_feedback = Feedback(
            feedback_type=feedback_type,
            prompt=feedback.prompt,
            prompt_metadata=feedback.prompt_metadata,
            annotator_id=feedback.annotator_id,
            confidence=feedback.confidence,
            justification=feedback.justification,
            tags=feedback.tags,
            model_id=feedback.model_id,
            model_version=feedback.model_version,
        )
        
        # Set type-specific fields
        if feedback.feedback_type == "preference":
            db_feedback.response_chosen = feedback.response_chosen
            db_feedback.response_rejected = feedback.response_rejected
        elif feedback.feedback_type == "score":
            db_feedback.response_text = feedback.response_text
            db_feedback.score = feedback.score
        elif feedback.feedback_type == "binary":
            db_feedback.response_text = feedback.response_text
            db_feedback.is_preferred = feedback.is_preferred
        elif feedback.feedback_type == "ranking":
            db_feedback.ranking = feedback.ranking
        
        session.add(db_feedback)
        await session.commit()
        await session.refresh(db_feedback)
        
        logger.info(f"Created feedback {db_feedback.id} (type: {feedback_type.value})")
        
        return FeedbackResponse.model_validate(db_feedback)
        
    except Exception as e:
        await session.rollback()
        logger.error(f"Failed to create feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create feedback: {str(e)}"
        )


@router.get(
    "",
    response_model=FeedbackListResponse,
    summary="List feedback",
    description="List feedback with filtering and pagination"
)
async def list_feedback(
    filters: FeedbackFilters = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """
    List feedback with filters.
    
    Supports filtering by:
    - feedback_type
    - annotator_id
    - task_id
    - dataset_id
    - model_id
    - min_confidence
    - created_after/before
    - tags
    
    Returns:
        Paginated list of feedback
    """
    try:
        # Build query
        query = select(Feedback)
        conditions = []
        
        if filters.feedback_type:
            conditions.append(Feedback.feedback_type == FeedbackType(filters.feedback_type.value))
        if filters.annotator_id:
            conditions.append(Feedback.annotator_id == filters.annotator_id)
        if filters.task_id:
            conditions.append(Feedback.task_id == filters.task_id)
        if filters.dataset_id:
            conditions.append(Feedback.dataset_id == filters.dataset_id)
        if filters.model_id:
            conditions.append(Feedback.model_id == filters.model_id)
        if filters.min_confidence is not None:
            conditions.append(Feedback.confidence >= filters.min_confidence)
        if filters.created_after:
            conditions.append(Feedback.created_at >= filters.created_after)
        if filters.created_before:
            conditions.append(Feedback.created_at <= filters.created_before)
        if filters.tags:
            # Check if any tag matches (PostgreSQL JSONB contains)
            for tag in filters.tags:
                conditions.append(Feedback.tags.contains([tag]))
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Count total
        count_query = select(func.count()).select_from(Feedback)
        if conditions:
            count_query = count_query.where(and_(*conditions))
        total_result = await session.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        offset = (filters.page - 1) * filters.page_size
        query = query.order_by(Feedback.created_at.desc())
        query = query.offset(offset).limit(filters.page_size)
        
        # Execute query
        result = await session.execute(query)
        feedback_items = result.scalars().all()
        
        # Check if has next page
        has_next = offset + len(feedback_items) < total
        
        return FeedbackListResponse(
            items=[FeedbackResponse.model_validate(f) for f in feedback_items],
            total=total,
            page=filters.page,
            page_size=filters.page_size,
            has_next=has_next,
        )
        
    except Exception as e:
        logger.error(f"Failed to list feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list feedback: {str(e)}"
        )


@router.get(
    "/{feedback_id}",
    response_model=FeedbackResponse,
    summary="Get feedback",
    description="Get specific feedback by ID"
)
async def get_feedback(
    feedback_id: int,
    session: AsyncSession = Depends(get_session),
):
    """
    Get feedback by ID.
    
    Args:
        feedback_id: Feedback ID
    
    Returns:
        Feedback details
    
    Raises:
        404: Feedback not found
    """
    try:
        result = await session.execute(
            select(Feedback).where(Feedback.id == feedback_id)
        )
        feedback = result.scalar_one_or_none()
        
        if not feedback:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Feedback {feedback_id} not found"
            )
        
        return FeedbackResponse.model_validate(feedback)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get feedback {feedback_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feedback: {str(e)}"
        )


@router.delete(
    "/{feedback_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete feedback",
    description="Delete specific feedback by ID"
)
async def delete_feedback(
    feedback_id: int,
    session: AsyncSession = Depends(get_session),
):
    """
    Delete feedback by ID.
    
    Args:
        feedback_id: Feedback ID
    
    Raises:
        404: Feedback not found
    """
    try:
        result = await session.execute(
            select(Feedback).where(Feedback.id == feedback_id)
        )
        feedback = result.scalar_one_or_none()
        
        if not feedback:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Feedback {feedback_id} not found"
            )
        
        await session.delete(feedback)
        await session.commit()
        
        logger.info(f"Deleted feedback {feedback_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Failed to delete feedback {feedback_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete feedback: {str(e)}"
        )
