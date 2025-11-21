"""RAG index and pipeline parsing."""

from __future__ import annotations
import re
from typing import TYPE_CHECKING, List, Optional, Dict, Any, Tuple

if TYPE_CHECKING:
    from .helpers import _Line

from namel3ss.ast.rag import IndexDefinition, RagPipelineDefinition

# Regex patterns for RAG declarations
_INDEX_HEADER_RE = re.compile(r'^index\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:{]')
_DATASET_HEADER_RE = re.compile(r'^dataset\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:{]')
_RAG_PIPELINE_HEADER_RE = re.compile(r'^rag_pipeline\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:{]')


class RAGParserMixin:
    """Mixin providing rag index and pipeline parsing."""
    
    def _parse_bool(self, value) -> bool:
        """Parse boolean value from various types."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1', 'on')
        return bool(value)

    def _parse_index(self, line: _Line) -> None:
        """
        Parse an index definition block.
        
        Grammar:
            index <name>:
                source_dataset: <dataset_name>
                embedding_model: <model_name>
                chunk_size: <int>
                overlap: <int>
                backend: <pgvector|qdrant|weaviate>
                namespace: <string>
                collection: <string>
                table_name: <string>
                metadata_fields: [<field1>, <field2>, ...]
        """
        match = _INDEX_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "index declaration")
        name = match.group(1)
        base_indent = self._indent(line.text)
        self._advance()
        
        # Parse key-value properties
        properties = self._parse_kv_block(base_indent)
        
        # Extract required fields
        source_dataset = properties.get('source_dataset')
        embedding_model = properties.get('embedding_model')
        if not source_dataset or not embedding_model:
            raise self._error("index block requires 'source_dataset' and 'embedding_model' fields", line)
        
        # Extract optional fields with defaults
        chunk_size = int(properties.get('chunk_size', 512))
        overlap = int(properties.get('overlap', 64))
        backend = properties.get('backend', 'pgvector')
        namespace = properties.get('namespace')
        collection = properties.get('collection')
        table_name = properties.get('table_name')
        
        # Multimodal fields
        extract_images = self._parse_bool(properties.get('extract_images', False))
        extract_audio = self._parse_bool(properties.get('extract_audio', False))
        image_model = properties.get('image_model')
        audio_model = properties.get('audio_model')
        
        # Parse metadata_fields if present (can be list or comma-separated string)
        metadata_fields = None
        if 'metadata_fields' in properties:
            mf = properties['metadata_fields']
            if isinstance(mf, list):
                metadata_fields = mf
            elif isinstance(mf, str):
                # Parse string like "[field1, field2]" or "field1, field2"
                mf = mf.strip()
                if mf.startswith('[') and mf.endswith(']'):
                    mf = mf[1:-1]
                metadata_fields = [f.strip() for f in mf.split(',') if f.strip()]
        
        # Build config from remaining properties
        config = {k: v for k, v in properties.items() 
                  if k not in {'source_dataset', 'embedding_model', 'chunk_size', 'overlap', 
                               'backend', 'namespace', 'collection', 'table_name', 'metadata_fields',
                               'extract_images', 'extract_audio', 'image_model', 'audio_model'}}
        
        index = IndexDefinition(
            name=name,
            source_dataset=source_dataset,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            overlap=overlap,
            backend=backend,
            namespace=namespace,
            collection=collection,
            table_name=table_name,
            metadata_fields=metadata_fields,
            extract_images=extract_images,
            extract_audio=extract_audio,
            image_model=image_model,
            audio_model=audio_model,
            config=config,
        )
        
        self._ensure_app(line)
        self._app.indices.append(index)

    def _parse_rag_pipeline(self, line: _Line) -> None:
        """
        Parse a RAG pipeline definition block.
        
        Grammar:
            rag_pipeline <name>:
                query_encoder: <embedding_model_name>
                index: <index_name>
                top_k: <int>
                reranker: <reranker_model_name>
                distance_metric: <cosine|euclidean|dot>
                filters: {<key>: <value>, ...}
        """
        match = _RAG_PIPELINE_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "rag_pipeline declaration")
        name = match.group(1)
        base_indent = self._indent(line.text)
        self._advance()
        
        # Parse key-value properties
        properties = self._parse_kv_block(base_indent)
        
        # Extract required fields
        query_encoder = properties.get('query_encoder')
        index = properties.get('index')
        if not query_encoder or not index:
            raise self._error("rag_pipeline block requires 'query_encoder' and 'index' fields", line)
        
        # Extract optional fields with defaults
        top_k = int(properties.get('top_k', 5))
        reranker = properties.get('reranker')
        distance_metric = properties.get('distance_metric', 'cosine')
        
        # Hybrid search fields
        enable_hybrid = self._parse_bool(properties.get('enable_hybrid', False))
        sparse_model = properties.get('sparse_model', 'bm25')
        dense_weight = float(properties.get('dense_weight', 0.7))
        sparse_weight = float(properties.get('sparse_weight', 0.3))
        reranker_type = properties.get('reranker_type', 'cross_encoder')
        
        # Parse filters if present
        filters = None
        if 'filters' in properties:
            f = properties['filters']
            if isinstance(f, dict):
                filters = f
            elif isinstance(f, str):
                # Try to parse as dict-like string
                import json
                try:
                    filters = json.loads(f)
                except json.JSONDecodeError:
                    pass
        
        # Build config from remaining properties
        config = {k: v for k, v in properties.items() 
                  if k not in {'query_encoder', 'index', 'top_k', 'reranker', 
                               'distance_metric', 'filters', 'enable_hybrid', 'sparse_model',
                               'dense_weight', 'sparse_weight', 'reranker_type'}}
        
        rag_pipeline = RagPipelineDefinition(
            name=name,
            query_encoder=query_encoder,
            index=index,
            top_k=top_k,
            reranker=reranker,
            distance_metric=distance_metric,
            filters=filters,
            enable_hybrid=enable_hybrid,
            sparse_model=sparse_model,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            reranker_type=reranker_type,
            config=config,
        )
        
        self._ensure_app(line)
        self._app.rag_pipelines.append(rag_pipeline)

    def _parse_agent(self, line: _Line) -> None:
        """Agent parser - moved to agents.py mixin."""
        pass

__all__ = ['RAGParserMixin']
