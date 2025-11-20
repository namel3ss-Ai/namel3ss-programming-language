"""Tests for RAG dataset loaders."""

import asyncio
import csv
import json
import pytest
from pathlib import Path
from typing import List

from namel3ss.rag.loaders import (
    LoadedDocument,
    CSVDatasetLoader,
    JSONDatasetLoader,
    InlineDatasetLoader,
    DatabaseDatasetLoader,
)


@pytest.fixture
def temp_csv_file(tmp_path):
    """Create a temporary CSV file for testing."""
    csv_file = tmp_path / "test_data.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "content", "tag", "author"])
        writer.writeheader()
        writer.writerow({
            "id": "doc1",
            "content": "First document content",
            "tag": "support",
            "author": "Alice"
        })
        writer.writerow({
            "id": "doc2",
            "content": "Second document content",
            "tag": "sales",
            "author": "Bob"
        })
        writer.writerow({
            "id": "doc3",
            "content": "Third document content",
            "tag": "support",
            "author": "Charlie"
        })
    return csv_file


@pytest.fixture
def temp_json_file(tmp_path):
    """Create a temporary JSON file for testing."""
    json_file = tmp_path / "test_data.json"
    data = [
        {"id": "doc1", "content": "First JSON document", "tag": "tech"},
        {"id": "doc2", "content": "Second JSON document", "tag": "business"},
        {"id": "doc3", "content": "Third JSON document", "tag": "tech"},
    ]
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return json_file


@pytest.fixture
def temp_jsonl_file(tmp_path):
    """Create a temporary JSONL file for testing."""
    jsonl_file = tmp_path / "test_data.jsonl"
    records = [
        {"id": "doc1", "content": "First JSONL document", "category": "A"},
        {"id": "doc2", "content": "Second JSONL document", "category": "B"},
        {"id": "doc3", "content": "Third JSONL document", "category": "A"},
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return jsonl_file


class TestCSVDatasetLoader:
    """Tests for CSV dataset loader."""
    
    @pytest.mark.asyncio
    async def test_load_all_documents(self, temp_csv_file):
        """Test loading all documents from CSV."""
        loader = CSVDatasetLoader(
            dataset_name="test_csv",
            file_path=temp_csv_file,
            content_field="content",
            id_field="id",
        )
        
        documents = []
        async for doc in loader.iter_documents():
            documents.append(doc)
        
        assert len(documents) == 3
        assert documents[0]["id"] == "doc1"
        assert documents[0]["content"] == "First document content"
        assert documents[0]["metadata"]["tag"] == "support"
        assert documents[0]["metadata"]["author"] == "Alice"
    
    @pytest.mark.asyncio
    async def test_load_with_limit(self, temp_csv_file):
        """Test loading documents with limit."""
        loader = CSVDatasetLoader(
            dataset_name="test_csv",
            file_path=temp_csv_file,
            content_field="content",
            id_field="id",
        )
        
        documents = []
        async for doc in loader.iter_documents(limit=2):
            documents.append(doc)
        
        assert len(documents) == 2
        assert documents[0]["id"] == "doc1"
        assert documents[1]["id"] == "doc2"
    
    @pytest.mark.asyncio
    async def test_load_with_filters(self, temp_csv_file):
        """Test loading documents with metadata filters."""
        loader = CSVDatasetLoader(
            dataset_name="test_csv",
            file_path=temp_csv_file,
            content_field="content",
            id_field="id",
        )
        
        documents = []
        async for doc in loader.iter_documents(filters={"tag": "support"}):
            documents.append(doc)
        
        assert len(documents) == 2
        assert all(doc["metadata"]["tag"] == "support" for doc in documents)
    
    @pytest.mark.asyncio
    async def test_load_with_offset(self, temp_csv_file):
        """Test loading documents with offset."""
        loader = CSVDatasetLoader(
            dataset_name="test_csv",
            file_path=temp_csv_file,
            content_field="content",
            id_field="id",
        )
        
        documents = []
        async for doc in loader.iter_documents(offset=1, limit=2):
            documents.append(doc)
        
        assert len(documents) == 2
        assert documents[0]["id"] == "doc2"
        assert documents[1]["id"] == "doc3"
    
    @pytest.mark.asyncio
    async def test_load_with_custom_delimiter(self, tmp_path):
        """Test loading CSV with custom delimiter."""
        tsv_file = tmp_path / "test_data.tsv"
        with open(tsv_file, "w", encoding="utf-8") as f:
            f.write("id\tcontent\ttag\n")
            f.write("doc1\tFirst document\tA\n")
            f.write("doc2\tSecond document\tB\n")
        
        loader = CSVDatasetLoader(
            dataset_name="test_tsv",
            file_path=tsv_file,
            content_field="content",
            id_field="id",
            config={"delimiter": "\t"},
        )
        
        documents = []
        async for doc in loader.iter_documents():
            documents.append(doc)
        
        assert len(documents) == 2
        assert documents[0]["content"] == "First document"
    
    @pytest.mark.asyncio
    async def test_missing_file(self, tmp_path):
        """Test handling missing CSV file."""
        missing_file = tmp_path / "missing.csv"
        loader = CSVDatasetLoader(
            dataset_name="test_csv",
            file_path=missing_file,
            content_field="content",
        )
        
        documents = []
        async for doc in loader.iter_documents():
            documents.append(doc)
        
        assert len(documents) == 0


class TestJSONDatasetLoader:
    """Tests for JSON dataset loader."""
    
    @pytest.mark.asyncio
    async def test_load_json_array(self, temp_json_file):
        """Test loading JSON array."""
        loader = JSONDatasetLoader(
            dataset_name="test_json",
            file_path=temp_json_file,
            content_field="content",
            id_field="id",
            is_jsonl=False,
        )
        
        documents = []
        async for doc in loader.iter_documents():
            documents.append(doc)
        
        assert len(documents) == 3
        assert documents[0]["id"] == "doc1"
        assert documents[0]["content"] == "First JSON document"
    
    @pytest.mark.asyncio
    async def test_load_jsonl(self, temp_jsonl_file):
        """Test loading line-delimited JSON."""
        loader = JSONDatasetLoader(
            dataset_name="test_jsonl",
            file_path=temp_jsonl_file,
            content_field="content",
            id_field="id",
            is_jsonl=True,
        )
        
        documents = []
        async for doc in loader.iter_documents():
            documents.append(doc)
        
        assert len(documents) == 3
        assert documents[0]["id"] == "doc1"
        assert documents[0]["content"] == "First JSONL document"
    
    @pytest.mark.asyncio
    async def test_load_with_filters(self, temp_json_file):
        """Test loading with filters."""
        loader = JSONDatasetLoader(
            dataset_name="test_json",
            file_path=temp_json_file,
            content_field="content",
            id_field="id",
        )
        
        documents = []
        async for doc in loader.iter_documents(filters={"tag": "tech"}):
            documents.append(doc)
        
        assert len(documents) == 2
        assert all(doc["metadata"]["tag"] == "tech" for doc in documents)
    
    @pytest.mark.asyncio
    async def test_invalid_json(self, tmp_path):
        """Test handling invalid JSON."""
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, "w", encoding="utf-8") as f:
            f.write("{invalid json}")
        
        loader = JSONDatasetLoader(
            dataset_name="test_json",
            file_path=invalid_file,
            content_field="content",
        )
        
        documents = []
        async for doc in loader.iter_documents():
            documents.append(doc)
        
        # Should handle error gracefully and return no documents
        assert len(documents) == 0


class TestInlineDatasetLoader:
    """Tests for inline dataset loader."""
    
    @pytest.mark.asyncio
    async def test_load_inline_records(self):
        """Test loading inline records."""
        records = [
            {"id": "doc1", "content": "First inline doc", "status": "active"},
            {"id": "doc2", "content": "Second inline doc", "status": "archived"},
            {"id": "doc3", "content": "Third inline doc", "status": "active"},
        ]
        
        loader = InlineDatasetLoader(
            dataset_name="test_inline",
            records=records,
            content_field="content",
            id_field="id",
        )
        
        documents = []
        async for doc in loader.iter_documents():
            documents.append(doc)
        
        assert len(documents) == 3
        assert documents[0]["id"] == "doc1"
        assert documents[0]["content"] == "First inline doc"
    
    @pytest.mark.asyncio
    async def test_empty_records(self):
        """Test loading from empty records list."""
        loader = InlineDatasetLoader(
            dataset_name="test_inline",
            records=[],
            content_field="content",
        )
        
        documents = []
        async for doc in loader.iter_documents():
            documents.append(doc)
        
        assert len(documents) == 0
    
    @pytest.mark.asyncio
    async def test_auto_generated_ids(self):
        """Test auto-generated document IDs."""
        records = [
            {"content": "Doc without ID 1"},
            {"content": "Doc without ID 2"},
        ]
        
        loader = InlineDatasetLoader(
            dataset_name="test_inline",
            records=records,
            content_field="content",
        )
        
        documents = []
        async for doc in loader.iter_documents():
            documents.append(doc)
        
        assert len(documents) == 2
        assert documents[0]["id"] == "test_inline_1"
        assert documents[1]["id"] == "test_inline_2"


class TestDatabaseDatasetLoader:
    """Tests for database dataset loader."""
    
    class MockConnector:
        """Mock database connector for testing."""
        
        def __init__(self, records: List[dict]):
            self.records = records
        
        async def execute_query(self, query: str, params: dict):
            """Mock query execution."""
            for record in self.records:
                yield record
    
    @pytest.mark.asyncio
    async def test_load_from_database(self):
        """Test loading documents from database."""
        records = [
            {"id": 1, "content": "Database doc 1", "status": "published"},
            {"id": 2, "content": "Database doc 2", "status": "draft"},
            {"id": 3, "content": "Database doc 3", "status": "published"},
        ]
        
        connector = self.MockConnector(records)
        loader = DatabaseDatasetLoader(
            dataset_name="test_db",
            connector=connector,
            query="SELECT * FROM documents",
            content_field="content",
            id_field="id",
        )
        
        documents = []
        async for doc in loader.iter_documents():
            documents.append(doc)
        
        assert len(documents) == 3
        assert documents[0]["id"] == "1"
        assert documents[0]["content"] == "Database doc 1"
    
    @pytest.mark.asyncio
    async def test_load_with_filters(self):
        """Test loading with metadata filters."""
        records = [
            {"id": 1, "content": "Database doc 1", "status": "published"},
            {"id": 2, "content": "Database doc 2", "status": "draft"},
            {"id": 3, "content": "Database doc 3", "status": "published"},
        ]
        
        connector = self.MockConnector(records)
        loader = DatabaseDatasetLoader(
            dataset_name="test_db",
            connector=connector,
            query="SELECT * FROM documents",
            content_field="content",
            id_field="id",
        )
        
        documents = []
        async for doc in loader.iter_documents(filters={"status": "published"}):
            documents.append(doc)
        
        assert len(documents) == 2
        assert all(doc["metadata"]["status"] == "published" for doc in documents)


class TestLoaderEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_empty_content_field(self, tmp_path):
        """Test handling records with empty content."""
        csv_file = tmp_path / "empty_content.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "content"])
            writer.writeheader()
            writer.writerow({"id": "doc1", "content": ""})
            writer.writerow({"id": "doc2", "content": "Valid content"})
        
        loader = CSVDatasetLoader(
            dataset_name="test_csv",
            file_path=csv_file,
            content_field="content",
            id_field="id",
        )
        
        documents = []
        async for doc in loader.iter_documents():
            documents.append(doc)
        
        # Should skip empty content
        assert len(documents) == 1
        assert documents[0]["id"] == "doc2"
    
    @pytest.mark.asyncio
    async def test_selected_metadata_fields(self, temp_csv_file):
        """Test loading only selected metadata fields."""
        loader = CSVDatasetLoader(
            dataset_name="test_csv",
            file_path=temp_csv_file,
            content_field="content",
            id_field="id",
            metadata_fields=["tag"],  # Only include 'tag', not 'author'
        )
        
        documents = []
        async for doc in loader.iter_documents(limit=1):
            documents.append(doc)
        
        assert len(documents) == 1
        assert "tag" in documents[0]["metadata"]
        assert "author" not in documents[0]["metadata"]
        assert "source" in documents[0]["metadata"]  # Always included
