"""
RLHF Datasets API - Endpoints for managing and exporting datasets.

Provides:
- POST /api/rlhf/datasets - Create dataset
- GET /api/rlhf/datasets - List datasets
- GET /api/rlhf/datasets/{id} - Get dataset
- POST /api/rlhf/datasets/{id}/export - Export dataset
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..models import Dataset, FeedbackType, DatasetStatus
from ..schemas import (
    DatasetCreate,
    DatasetResponse,
    DatasetListResponse,
    DatasetFilters,
)
from ..exporters import DatasetExporter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rlhf/datasets", tags=["datasets"])


@router.post(
    "",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create dataset",
    description="Create dataset for export"
)
async def create_dataset(
    dataset: DatasetCreate,
    session: AsyncSession = Depends(get_session),
):
    """
    Create dataset configuration.
    
    Args:
        dataset: Dataset configuration
    
    Returns:
        Created dataset
    """
    try:
        # Check if name already exists
        result = await session.execute(
            select(Dataset).where(Dataset.name == dataset.name)
        )
        existing = result.scalar_one_or_none()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset with name '{dataset.name}' already exists"
            )
        
        feedback_type = FeedbackType(dataset.feedback_type.value)
        
        db_dataset = Dataset(
            name=dataset.name,
            description=dataset.description,
            feedback_type=feedback_type,
            export_format=dataset.export_format,
            min_confidence=dataset.min_confidence,
            annotator_filter=dataset.annotator_filter,
            date_range_start=dataset.date_range_start,
            date_range_end=dataset.date_range_end,
            metadata=dataset.metadata,
        )
        
        session.add(db_dataset)
        await session.commit()
        await session.refresh(db_dataset)
        
        logger.info(f"Created dataset {db_dataset.id}: {db_dataset.name}")
        
        return DatasetResponse.model_validate(db_dataset)
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Failed to create dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create dataset: {str(e)}"
        )


@router.get(
    "",
    response_model=DatasetListResponse,
    summary="List datasets",
    description="List datasets with filtering and pagination"
)
async def list_datasets(
    filters: DatasetFilters = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """
    List datasets with filters.
    
    Returns:
        Paginated list of datasets
    """
    try:
        query = select(Dataset)
        conditions = []
        
        if filters.status:
            conditions.append(Dataset.status == DatasetStatus(filters.status.value))
        if filters.feedback_type:
            conditions.append(Dataset.feedback_type == FeedbackType(filters.feedback_type.value))
        if filters.export_format:
            conditions.append(Dataset.export_format == filters.export_format)
        if filters.created_after:
            conditions.append(Dataset.created_at >= filters.created_after)
        if filters.created_before:
            conditions.append(Dataset.created_at <= filters.created_before)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Count total
        count_query = select(func.count()).select_from(Dataset)
        if conditions:
            count_query = count_query.where(and_(*conditions))
        total_result = await session.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        offset = (filters.page - 1) * filters.page_size
        query = query.order_by(Dataset.created_at.desc())
        query = query.offset(offset).limit(filters.page_size)
        
        result = await session.execute(query)
        datasets = result.scalars().all()
        
        has_next = offset + len(datasets) < total
        
        return DatasetListResponse(
            items=[DatasetResponse.model_validate(d) for d in datasets],
            total=total,
            page=filters.page,
            page_size=filters.page_size,
            has_next=has_next,
        )
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list datasets: {str(e)}"
        )


@router.get(
    "/{dataset_id}",
    response_model=DatasetResponse,
    summary="Get dataset",
    description="Get specific dataset by ID"
)
async def get_dataset(
    dataset_id: int,
    session: AsyncSession = Depends(get_session),
):
    """
    Get dataset by ID.
    
    Args:
        dataset_id: Dataset ID
    
    Returns:
        Dataset details
    """
    try:
        result = await session.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        dataset = result.scalar_one_or_none()
        
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        
        return DatasetResponse.model_validate(dataset)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset {dataset_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dataset: {str(e)}"
        )


@router.post(
    "/{dataset_id}/export",
    response_model=Dict[str, Any],
    summary="Export dataset",
    description="Export dataset to file"
)
async def export_dataset(
    dataset_id: int,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    """
    Export dataset to file.
    
    Args:
        dataset_id: Dataset ID
        background_tasks: Background task queue
    
    Returns:
        Export job details
    """
    try:
        result = await session.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        dataset = result.scalar_one_or_none()
        
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )
        
        if dataset.status == DatasetStatus.PROCESSING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset export already in progress"
            )
        
        # Determine output path
        if dataset.export_format == "hf":
            # HuggingFace repo ID
            output_path = f"rlhf-datasets/{dataset.name}"
        else:
            # Local or S3 path
            output_path = f"rlhf_datasets/{dataset.name}"
        
        # Export in background
        exporter = DatasetExporter(session)
        background_tasks.add_task(
            exporter.export_dataset,
            dataset,
            output_path,
        )
        
        logger.info(f"Started export for dataset {dataset_id}")
        
        return {
            "dataset_id": dataset_id,
            "status": "processing",
            "message": "Dataset export started",
            "output_path": output_path,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export dataset {dataset_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export dataset: {str(e)}"
        )
