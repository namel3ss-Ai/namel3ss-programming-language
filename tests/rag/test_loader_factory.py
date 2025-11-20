"""Tests for dataset loader factory."""

import pytest
from pathlib import Path
from unittest.mock import Mock

from namel3ss.ast.datasets import Dataset, DatasetConnectorConfig
from namel3ss.rag.loader_factory import (
    get_dataset_loader,
    DatasetLoaderError,
)
from namel3ss.rag.loaders import (
    CSVDatasetLoader,
    JSONDatasetLoader,
    InlineDatasetLoader,
    DatabaseDatasetLoader,
)


@pytest.fixture
def csv_dataset(tmp_path):
    """Create a CSV dataset fixture."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("id,content\ndoc1,Test content\n")
    
    return Dataset(
        name="test_csv",
        source_type="csv",
        source=str(csv_file),
        metadata={"content_field": "content", "id_field": "id"},
    )


@pytest.fixture
def json_dataset(tmp_path):
    """Create a JSON dataset fixture."""
    json_file = tmp_path / "test.json"
    json_file.write_text('[{"id": "doc1", "content": "Test"}]')
    
    return Dataset(
        name="test_json",
        source_type="json",
        source=str(json_file),
        metadata={"content_field": "content", "id_field": "id"},
    )


@pytest.fixture
def jsonl_dataset(tmp_path):
    """Create a JSONL dataset fixture."""
    jsonl_file = tmp_path / "test.jsonl"
    jsonl_file.write_text('{"id": "doc1", "content": "Test"}\n')
    
    return Dataset(
        name="test_jsonl",
        source_type="jsonl",
        source=str(jsonl_file),
        metadata={"content_field": "content", "id_field": "id"},
    )


@pytest.fixture
def inline_dataset():
    """Create an inline dataset fixture."""
    return Dataset(
        name="test_inline",
        source_type="inline",
        source="",
        metadata={
            "content_field": "content",
            "records": [
                {"id": "doc1", "content": "Test content 1"},
                {"id": "doc2", "content": "Test content 2"},
            ]
        },
    )


class TestLoaderFactory:
    """Tests for get_dataset_loader factory function."""
    
    def test_create_csv_loader(self, csv_dataset):
        """Test creating a CSV loader from dataset definition."""
        loader = get_dataset_loader(csv_dataset)
        
        assert isinstance(loader, CSVDatasetLoader)
        assert loader.dataset_name == "test_csv"
        assert loader.content_field == "content"
        assert loader.id_field == "id"
    
    def test_create_json_loader(self, json_dataset):
        """Test creating a JSON loader from dataset definition."""
        loader = get_dataset_loader(json_dataset)
        
        assert isinstance(loader, JSONDatasetLoader)
        assert loader.dataset_name == "test_json"
        assert loader.is_jsonl is False
    
    def test_create_jsonl_loader(self, jsonl_dataset):
        """Test creating a JSONL loader from dataset definition."""
        loader = get_dataset_loader(jsonl_dataset)
        
        assert isinstance(loader, JSONDatasetLoader)
        assert loader.dataset_name == "test_jsonl"
        assert loader.is_jsonl is True
    
    def test_create_inline_loader(self, inline_dataset):
        """Test creating an inline loader from dataset definition."""
        loader = get_dataset_loader(inline_dataset)
        
        assert isinstance(loader, InlineDatasetLoader)
        assert loader.dataset_name == "test_inline"
        assert len(loader.records) == 2
    
    def test_unsupported_source_type(self):
        """Test error on unsupported source type."""
        dataset = Dataset(
            name="test_unsupported",
            source_type="unsupported_type",
            source="test.txt",
        )
        
        with pytest.raises(DatasetLoaderError) as exc_info:
            get_dataset_loader(dataset)
        
        assert "Unsupported dataset source_type" in str(exc_info.value)
    
    def test_csv_with_custom_delimiter(self, tmp_path):
        """Test CSV loader with custom delimiter."""
        tsv_file = tmp_path / "test.tsv"
        tsv_file.write_text("id\tcontent\ndoc1\tTest\n")
        
        dataset = Dataset(
            name="test_tsv",
            source_type="csv",
            source=str(tsv_file),
            metadata={
                "content_field": "content",
                "delimiter": "\t",
            },
        )
        
        loader = get_dataset_loader(dataset)
        assert isinstance(loader, CSVDatasetLoader)
        assert loader.config.get("delimiter") == "\t"
    
    def test_file_source_type_detection(self, tmp_path):
        """Test automatic detection based on file extension."""
        # CSV file with 'file' source_type
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,content\ndoc1,Test\n")
        
        dataset = Dataset(
            name="test_file",
            source_type="file",
            source=str(csv_file),
        )
        
        loader = get_dataset_loader(dataset)
        assert isinstance(loader, CSVDatasetLoader)
        
        # JSON file with 'file' source_type
        json_file = tmp_path / "data.json"
        json_file.write_text('[{"id": "doc1", "content": "Test"}]')
        
        dataset = Dataset(
            name="test_file",
            source_type="file",
            source=str(json_file),
        )
        
        loader = get_dataset_loader(dataset)
        assert isinstance(loader, JSONDatasetLoader)
    
    def test_metadata_field_mapping(self, csv_dataset):
        """Test that metadata fields are correctly passed to loader."""
        csv_dataset.metadata["metadata_fields"] = ["tag", "author"]
        
        loader = get_dataset_loader(csv_dataset)
        assert loader.metadata_fields == ["tag", "author"]
    
    def test_custom_connector_import_error(self):
        """Test error handling for invalid custom connector."""
        dataset = Dataset(
            name="test_custom",
            source_type="custom",
            source="",
            connector=DatasetConnectorConfig(
                connector_type="nonexistent.module.Loader",
                options={},
            ),
        )
        
        with pytest.raises(DatasetLoaderError) as exc_info:
            get_dataset_loader(dataset)
        
        assert "Failed to import custom loader" in str(exc_info.value)


class TestLoaderFactoryIntegration:
    """Integration tests for loader factory with real data."""
    
    @pytest.mark.asyncio
    async def test_csv_loader_end_to_end(self, tmp_path):
        """Test CSV loader from factory through document iteration."""
        csv_file = tmp_path / "products.csv"
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("id,name,description,category\n")
            f.write("p1,Product 1,Description 1,electronics\n")
            f.write("p2,Product 2,Description 2,books\n")
            f.write("p3,Product 3,Description 3,electronics\n")
        
        dataset = Dataset(
            name="products",
            source_type="csv",
            source=str(csv_file),
            metadata={
                "content_field": "description",
                "id_field": "id",
                "metadata_fields": ["name", "category"],
            },
        )
        
        loader = get_dataset_loader(dataset)
        
        documents = []
        async for doc in loader.iter_documents(filters={"category": "electronics"}):
            documents.append(doc)
        
        assert len(documents) == 2
        assert documents[0]["id"] == "p1"
        assert documents[0]["content"] == "Description 1"
        assert documents[0]["metadata"]["name"] == "Product 1"
        assert documents[0]["metadata"]["category"] == "electronics"
    
    @pytest.mark.asyncio
    async def test_jsonl_loader_end_to_end(self, tmp_path):
        """Test JSONL loader from factory through document iteration."""
        jsonl_file = tmp_path / "articles.jsonl"
        with open(jsonl_file, "w", encoding="utf-8") as f:
            f.write('{"id": "a1", "title": "Article 1", "body": "Content 1", "published": true}\n')
            f.write('{"id": "a2", "title": "Article 2", "body": "Content 2", "published": false}\n')
            f.write('{"id": "a3", "title": "Article 3", "body": "Content 3", "published": true}\n')
        
        dataset = Dataset(
            name="articles",
            source_type="jsonl",
            source=str(jsonl_file),
            metadata={
                "content_field": "body",
                "id_field": "id",
            },
        )
        
        loader = get_dataset_loader(dataset)
        
        documents = []
        async for doc in loader.iter_documents(limit=2):
            documents.append(doc)
        
        assert len(documents) == 2
        assert documents[0]["id"] == "a1"
        assert documents[0]["content"] == "Content 1"
    
    @pytest.mark.asyncio
    async def test_inline_loader_end_to_end(self):
        """Test inline loader from factory through document iteration."""
        dataset = Dataset(
            name="faq",
            source_type="inline",
            source="",
            metadata={
                "content_field": "answer",
                "id_field": "question_id",
                "records": [
                    {"question_id": "q1", "question": "What is X?", "answer": "X is...", "category": "general"},
                    {"question_id": "q2", "question": "How to Y?", "answer": "To Y, you...", "category": "howto"},
                ],
            },
        )
        
        loader = get_dataset_loader(dataset)
        
        documents = []
        async for doc in loader.iter_documents():
            documents.append(doc)
        
        assert len(documents) == 2
        assert documents[0]["id"] == "q1"
        assert documents[0]["content"] == "X is..."
        assert documents[0]["metadata"]["question"] == "What is X?"
