"""
Feature Detection from IR

This module analyzes Namel3ss IR to detect which features are actually used
in a project, enabling intelligent dependency generation.

Detection Strategy:
------------------
1. Parse .ai files to IR
2. Traverse IR nodes to find feature usage
3. Return set of feature IDs

Features Detected:
-----------------
- AI providers (openai, anthropic, ollama, vllm)
- UI components (chat, file_upload, chart, data_table, form, etc.)
- Database usage (postgres, mysql, mongo)
- Real-time features (websockets, redis)
- Observability (otel)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Set, List, Optional
import json

from namel3ss.parser import Parser
from namel3ss.ir.builder import build_backend_ir, build_frontend_ir
from namel3ss.ast.datasets import Dataset


@dataclass
class DetectedFeatures:
    """Result of feature detection"""
    features: Set[str] = field(default_factory=set)
    warnings: List[str] = field(default_factory=list)
    
    def add_feature(self, feature_id: str) -> None:
        """Add a detected feature"""
        self.features.add(feature_id)
    
    def add_warning(self, message: str) -> None:
        """Add a warning message"""
        self.warnings.append(message)
    
    def merge(self, other: 'DetectedFeatures') -> None:
        """Merge another DetectedFeatures into this one"""
        self.features.update(other.features)
        self.warnings.extend(other.warnings)


class FeatureDetector:
    """
    Detects features used in Namel3ss projects by analyzing IR.
    
    Usage:
        detector = FeatureDetector()
        features = detector.detect_from_file('app.ai')
        print(features.features)  # {'openai', 'chat', 'postgres', ...}
    """
    
    def __init__(self):
        self.detected = DetectedFeatures()
    
    def detect_from_file(self, file_path: str | Path) -> DetectedFeatures:
        """
        Detect features from a single .ai file.
        
        Args:
            file_path: Path to .ai file
            
        Returns:
            DetectedFeatures with feature IDs
        """
        path = Path(file_path)
        if not path.exists():
            self.detected.add_warning(f"File not found: {file_path}")
            return self.detected
        
        content = path.read_text(encoding='utf-8')
        return self.detect_from_source(content)
    
    def detect_from_source(self, source: str) -> DetectedFeatures:
        """
        Detect features from source code.
        
        Args:
            source: Namel3ss source code
            
        Returns:
            DetectedFeatures with feature IDs
        """
        self.detected = DetectedFeatures()
        
        # Always do text-based detection first (works even if parsing fails)
        source_lower = source.lower()
        if 'agent ' in source_lower or 'agent{' in source_lower:
            # Found agent definition
            self.detected.add_feature('openai')
        if 'from postgres' in source_lower or 'postgres table' in source_lower:
            self.detected.add_feature('postgres')
            self.detected.add_feature('sql')
        if 'from mysql' in source_lower or 'mysql table' in source_lower:
            self.detected.add_feature('mysql')
            self.detected.add_feature('sql')
        
        # Try AST-based detection for more accurate results
        try:
            # Parse to AST
            parser = Parser(source)
            module = parser.parse()
            
            # Check module-level items
            for item in module.body:
                if isinstance(item, Dataset):
                    self._detect_dataset_features(item)
                elif hasattr(item, '__class__') and item.__class__.__name__ == 'App':
                    # Process App
                    if hasattr(item, 'datasets'):
                        for ds in item.datasets:
                            self._detect_dataset_features(ds)
                    if hasattr(item, 'llms'):
                        self._detect_llm_features(item.llms)
                    if hasattr(item, 'agents') and item.agents:
                        self._detect_agent_features(item.agents, getattr(item, 'llms', []))
                        
        except Exception as e:
            # Text-based detection already ran, so we have basic features
            self.detected.add_warning(f"Error parsing file: {e}")
        
        return self.detected
    
    def _detect_dataset_features(self, dataset):
        """Detect features from a dataset"""
        if hasattr(dataset, 'source_type'):
            source_type = str(dataset.source_type).lower()
            if source_type == 'table' or 'table' in source_type:
                self.detected.add_feature('sql')
        
        if hasattr(dataset, 'source'):
            source = str(dataset.source).lower()
            if 'postgres' in source or 'pg' in source:
                self.detected.add_feature('postgres')
                self.detected.add_feature('sql')
            elif 'mysql' in source:
                self.detected.add_feature('mysql')
                self.detected.add_feature('sql')
            elif 'mongo' in source:
                self.detected.add_feature('mongo')
            elif 'table' in source:
                self.detected.add_feature('sql')
    
    def _detect_llm_features(self, llms):
        """Detect features from LLMs"""
        for llm in llms:
            if hasattr(llm, 'provider'):
                provider = str(llm.provider).lower() if llm.provider else 'openai'
                if 'openai' in provider or 'gpt' in provider:
                    self.detected.add_feature('openai')
                elif 'anthropic' in provider or 'claude' in provider:
                    self.detected.add_feature('anthropic')
                elif 'ollama' in provider:
                    self.detected.add_feature('ollama')
                elif 'vllm' in provider:
                    self.detected.add_feature('vllm')
            else:
                self.detected.add_feature('openai')  # Default
    
    def _detect_agent_features(self, agents, llms):
        """Detect features from agents"""
        if agents and len(agents) > 0:
            # If agents exist, assume OpenAI by default
            self.detected.add_feature('openai')
    
    def detect_from_directory(self, dir_path: str | Path) -> DetectedFeatures:
        """
        Detect features from all .ai files in a directory.
        
        Args:
            dir_path: Path to directory
            
        Returns:
            DetectedFeatures aggregated from all files
        """
        path = Path(dir_path)
        combined = DetectedFeatures()
        
        # Find all .ai files
        ai_files = list(path.rglob("*.ai"))
        
        if not ai_files:
            combined.add_warning(f"No .ai files found in {dir_path}")
            return combined
        
        for ai_file in ai_files:
            try:
                features = self.detect_from_file(ai_file)
                combined.merge(features)
            except Exception as e:
                combined.add_warning(f"Error processing {ai_file}: {e}")
        
        return combined
    
    def _detect_backend_features(self, backend_ir) -> None:
        """Detect backend features from IR"""
        # Check for database usage
        if hasattr(backend_ir, 'datasets') and backend_ir.datasets:
            for dataset in backend_ir.datasets:
                # Detect database type from source
                if hasattr(dataset, 'source'):
                    source = str(dataset.source).lower()
                    if 'postgres' in source or 'pg' in source:
                        self.detected.add_feature('postgres')
                        self.detected.add_feature('sql')
                    elif 'mysql' in source:
                        self.detected.add_feature('mysql')
                        self.detected.add_feature('sql')
                    elif 'mongo' in source:
                        self.detected.add_feature('mongo')
                    else:
                        # Default to SQL if table source
                        if 'table' in source:
                            self.detected.add_feature('sql')
        
        # Check for agents (indicates AI usage)
        if hasattr(backend_ir, 'agents') and backend_ir.agents:
            self.detected.add_feature('openai')  # Default assumption
        
        # Check for tools
        if hasattr(backend_ir, 'tools') and backend_ir.tools:
            pass  # Tools don't require specific deps beyond core
        
        # Check for memory systems
        if hasattr(backend_ir, 'memories') and backend_ir.memories:
            # Memory might need Redis for persistence
            for memory in backend_ir.memories:
                if hasattr(memory, 'type') and memory.type == 'long_term':
                    self.detected.add_feature('redis')
    
    def _detect_frontend_features(self, frontend_ir) -> None:
        """Detect frontend features from IR"""
        if not hasattr(frontend_ir, 'pages'):
            return
        
        for page in frontend_ir.pages:
            if not hasattr(page, 'components'):
                continue
            
            for component in page.components:
                component_type = component.__class__.__name__.lower()
                
                # Map component types to features
                if 'chat' in component_type or 'message' in component_type:
                    self.detected.add_feature('chat')
                elif 'upload' in component_type or 'file' in component_type:
                    self.detected.add_feature('file_upload')
                elif 'chart' in component_type or 'graph' in component_type:
                    self.detected.add_feature('chart')
                elif 'table' in component_type or 'grid' in component_type:
                    self.detected.add_feature('data_table')
                elif 'form' in component_type or 'input' in component_type:
                    self.detected.add_feature('form')
                elif 'markdown' in component_type:
                    self.detected.add_feature('markdown')
                elif 'code' in component_type and 'editor' in component_type:
                    self.detected.add_feature('code_editor')
    
    def _detect_from_ast(self, app) -> None:
        """Detect features from AST nodes"""
        # Check LLMs for provider detection
        if hasattr(app, 'llms'):
            for llm in app.llms:
                if hasattr(llm, 'provider'):
                    provider = llm.provider.lower() if llm.provider else 'openai'
                    if 'openai' in provider or 'gpt' in provider:
                        self.detected.add_feature('openai')
                    elif 'anthropic' in provider or 'claude' in provider:
                        self.detected.add_feature('anthropic')
                    elif 'ollama' in provider:
                        self.detected.add_feature('ollama')
                    elif 'vllm' in provider:
                        self.detected.add_feature('vllm')
                else:
                    # Default to OpenAI if no provider specified
                    self.detected.add_feature('openai')
        
        # Check agents
        if hasattr(app, 'agents'):
            for agent in app.agents:
                # If agent exists, we need AI provider
                if hasattr(agent, 'llm_name'):
                    # Try to find which LLM is used
                    if hasattr(app, 'llms'):
                        for llm in app.llms:
                            if llm.name == agent.llm_name:
                                # Already detected above
                                break
                    else:
                        # Default to OpenAI
                        self.detected.add_feature('openai')
        
        # Check for WebSocket/real-time features
        if hasattr(app, 'pages'):
            for page in app.pages:
                # Check page configuration for real-time
                if hasattr(page, 'realtime') and page.realtime:
                    self.detected.add_feature('websockets')
        
        # Check for observability configuration
        if hasattr(app, 'config'):
            config = app.config
            if isinstance(config, dict):
                if 'observability' in config or 'otel' in config:
                    self.detected.add_feature('otel')


def detect_features(source_or_path: str | Path) -> DetectedFeatures:
    """
    Convenience function to detect features from source or file.
    
    Args:
        source_or_path: Source code string or path to .ai file
        
    Returns:
        DetectedFeatures with feature IDs
    """
    detector = FeatureDetector()
    
    # Check if it's a path
    if isinstance(source_or_path, (str, Path)):
        path = Path(source_or_path)
        if path.exists():
            if path.is_file():
                return detector.detect_from_file(path)
            elif path.is_dir():
                return detector.detect_from_directory(path)
    
    # Assume it's source code
    return detector.detect_from_source(str(source_or_path))
