"""
RLHF Dataset Exporters - Export feedback to training datasets.

Provides:
- Export to Parquet files
- Export to JSONL files
- Export to HuggingFace Hub
- Data transformation and validation
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Feedback, Dataset, FeedbackType, DatasetStatus
from .errors import RLHFDatasetError

logger = logging.getLogger(__name__)


class DatasetExporter:
    """
    Export feedback to training datasets.
    
    Transforms feedback into format suitable for RLHF training:
    - Preference: (prompt, chosen, rejected) pairs
    - Score: (prompt, response, score) tuples
    - Binary: (prompt, response, label) tuples
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize exporter.
        
        Args:
            session: Database session
        """
        self.session = session
    
    async def export_dataset(
        self,
        dataset: Dataset,
        output_path: str,
    ) -> Dict[str, Any]:
        """
        Export dataset to file.
        
        Args:
            dataset: Dataset to export
            output_path: Output path (file:// or s3://)
        
        Returns:
            Export statistics
        
        Raises:
            RLHFDatasetError: If export fails
        """
        try:
            # Mark as processing
            dataset.status = DatasetStatus.PROCESSING
            await self.session.commit()
            
            # Query feedback
            feedback_items = await self._query_feedback(dataset)
            
            if not feedback_items:
                raise RLHFDatasetError(
                    "No feedback items found for dataset",
                    error_code="RLHF030",
                )
            
            # Transform to training format
            df = await self._transform_feedback(feedback_items, dataset.feedback_type)
            
            # Split into train/val/test
            train_df, val_df, test_df = self._split_dataset(df, dataset)
            
            # Export based on format
            if dataset.export_format == "parquet":
                stats = await self._export_parquet(train_df, val_df, test_df, output_path)
            elif dataset.export_format == "jsonl":
                stats = await self._export_jsonl(train_df, val_df, test_df, output_path)
            elif dataset.export_format == "hf":
                stats = await self._export_huggingface(train_df, val_df, test_df, output_path, dataset)
            else:
                raise RLHFDatasetError(
                    f"Unsupported export format: {dataset.export_format}",
                    error_code="RLHF030",
                )
            
            # Update dataset
            dataset.status = DatasetStatus.READY
            dataset.export_path = output_path
            dataset.num_samples = len(df)
            dataset.train_samples = len(train_df)
            dataset.val_samples = len(val_df)
            dataset.test_samples = len(test_df)
            await self.session.commit()
            
            logger.info(f"Exported dataset {dataset.id} to {output_path}")
            
            return stats
            
        except Exception as e:
            dataset.status = DatasetStatus.FAILED
            await self.session.commit()
            logger.error(f"Failed to export dataset {dataset.id}: {e}")
            raise RLHFDatasetError(
                f"Dataset export failed: {e}",
                error_code="RLHF030",
            ) from e
    
    async def _query_feedback(self, dataset: Dataset) -> List[Feedback]:
        """Query feedback items based on dataset filters."""
        query = select(Feedback).where(
            Feedback.feedback_type == dataset.feedback_type
        )
        
        conditions = []
        
        # Apply filters
        if dataset.min_confidence is not None:
            conditions.append(Feedback.confidence >= dataset.min_confidence)
        
        if dataset.annotator_filter:
            conditions.append(Feedback.annotator_id.in_(dataset.annotator_filter))
        
        if dataset.date_range_start:
            conditions.append(Feedback.created_at >= dataset.date_range_start)
        
        if dataset.date_range_end:
            conditions.append(Feedback.created_at <= dataset.date_range_end)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.order_by(Feedback.created_at)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def _transform_feedback(
        self,
        feedback_items: List[Feedback],
        feedback_type: FeedbackType,
    ) -> pd.DataFrame:
        """Transform feedback to training format."""
        data = []
        
        for item in feedback_items:
            row = {
                "prompt": item.prompt,
                "feedback_id": item.id,
                "annotator_id": item.annotator_id,
                "confidence": item.confidence,
                "created_at": item.created_at,
            }
            
            if feedback_type == FeedbackType.PREFERENCE:
                row.update({
                    "chosen": item.response_chosen,
                    "rejected": item.response_rejected,
                })
            elif feedback_type == FeedbackType.SCORE:
                row.update({
                    "response": item.response_text,
                    "score": item.score,
                })
            elif feedback_type == FeedbackType.BINARY:
                row.update({
                    "response": item.response_text,
                    "label": 1 if item.is_preferred else 0,
                })
            elif feedback_type == FeedbackType.RANKING:
                row.update({
                    "ranking": item.ranking,
                })
            
            # Add metadata
            if item.prompt_metadata:
                row["prompt_metadata"] = item.prompt_metadata
            if item.tags:
                row["tags"] = item.tags
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _split_dataset(
        self,
        df: pd.DataFrame,
        dataset: Dataset,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/val/test."""
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split sizes
        total = len(df)
        train_size = int(total * 0.8)  # Default 80/10/10
        val_size = int(total * 0.1)
        
        # Split
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        return train_df, val_df, test_df
    
    async def _export_parquet(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_path: str,
    ) -> Dict[str, Any]:
        """Export to Parquet files."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = output_dir / "train.parquet"
        val_path = output_dir / "val.parquet"
        test_path = output_dir / "test.parquet"
        
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)
        
        return {
            "format": "parquet",
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
        }
    
    async def _export_jsonl(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_path: str,
    ) -> Dict[str, Any]:
        """Export to JSONL files."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = output_dir / "train.jsonl"
        val_path = output_dir / "val.jsonl"
        test_path = output_dir / "test.jsonl"
        
        train_df.to_json(train_path, orient="records", lines=True)
        val_df.to_json(val_path, orient="records", lines=True)
        test_df.to_json(test_path, orient="records", lines=True)
        
        return {
            "format": "jsonl",
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
        }
    
    async def _export_huggingface(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        repo_id: str,
        dataset: Dataset,
    ) -> Dict[str, Any]:
        """Export to HuggingFace Hub."""
        try:
            from datasets import Dataset as HFDataset, DatasetDict
            
            # Convert to HF datasets
            hf_train = HFDataset.from_pandas(train_df)
            hf_val = HFDataset.from_pandas(val_df)
            hf_test = HFDataset.from_pandas(test_df)
            
            # Create dataset dict
            dataset_dict = DatasetDict({
                "train": hf_train,
                "validation": hf_val,
                "test": hf_test,
            })
            
            # Push to hub
            dataset_dict.push_to_hub(
                repo_id,
                private=True,  # Default to private
            )
            
            return {
                "format": "huggingface",
                "repo_id": repo_id,
                "url": f"https://huggingface.co/datasets/{repo_id}",
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
            }
            
        except ImportError:
            raise RLHFDatasetError(
                "HuggingFace datasets library not installed. Install with: pip install datasets",
                error_code="RLHF030",
            )
        except Exception as e:
            raise RLHFDatasetError(
                f"Failed to export to HuggingFace Hub: {e}",
                error_code="RLHF030",
            ) from e
