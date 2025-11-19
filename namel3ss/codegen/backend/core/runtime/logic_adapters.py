"""
Adapters for exposing datasets and models as logic facts.

Provides a well-defined bridge between N3's data layer (datasets, model registry)
and the logic engine. No ad-hoc introspection - everything is explicitly typed
and validated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

from namel3ss.ast.logic import LogicAtom, LogicFact, LogicNumber, LogicString, LogicStruct


# ============================================================================
# Base Adapter Interface
# ============================================================================

class LogicAdapter(ABC):
    """
    Base class for adapters that expose data as logic facts.
    
    Adapters provide a clean, typed interface for converting application data
    into facts that can be queried by the logic engine.
    """
    
    @abstractmethod
    def get_predicates(self) -> List[str]:
        """Return the list of predicate names this adapter provides."""
        pass
    
    @abstractmethod
    def get_facts(self, predicate: Optional[str] = None) -> Iterable[LogicFact]:
        """
        Generate facts for the given predicate.
        
        If predicate is None, generate all facts.
        Facts are generated lazily for efficiency.
        """
        pass
    
    @abstractmethod
    def get_arity(self, predicate: str) -> int:
        """Return the arity (number of arguments) for a predicate."""
        pass


# ============================================================================
# Dataset Adapter
# ============================================================================

class DatasetAdapter(LogicAdapter):
    """
    Adapter for exposing dataset records as logic facts.
    
    Maps dataset rows to predicates like:
    - row(dataset_name, row_id, field1, field2, ...)
    - field(dataset_name, row_id, field_name, field_value)
    
    This provides a flexible way to query dataset contents without
    requiring ad-hoc introspection.
    """
    
    def __init__(self, dataset_name: str, schema: Dict[str, str], records: List[Dict[str, Any]]):
        """
        Initialize dataset adapter.
        
        Args:
            dataset_name: Name of the dataset
            schema: Field name -> type mapping
            records: List of record dictionaries
        """
        self.dataset_name = dataset_name
        self.schema = schema
        self.records = records
        self._validate_records()
    
    def _validate_records(self) -> None:
        """Validate that records conform to schema."""
        for i, record in enumerate(self.records):
            for field_name in self.schema.keys():
                if field_name not in record:
                    raise ValueError(
                        f"Record {i} missing required field: {field_name}"
                    )
    
    def get_predicates(self) -> List[str]:
        """Return predicates: row/N, field/4."""
        return [
            f"row_{self.dataset_name}",
            f"field_{self.dataset_name}",
        ]
    
    def get_arity(self, predicate: str) -> int:
        """Return predicate arity."""
        if predicate == f"row_{self.dataset_name}":
            return 1 + len(self.schema)  # row_id + fields
        elif predicate == f"field_{self.dataset_name}":
            return 3  # row_id, field_name, field_value
        else:
            raise ValueError(f"Unknown predicate: {predicate}")
    
    def get_facts(self, predicate: Optional[str] = None) -> Iterable[LogicFact]:
        """Generate facts for dataset records."""
        if predicate is None or predicate == f"row_{self.dataset_name}":
            yield from self._generate_row_facts()
        
        if predicate is None or predicate == f"field_{self.dataset_name}":
            yield from self._generate_field_facts()
    
    def _generate_row_facts(self) -> Iterable[LogicFact]:
        """
        Generate row facts: row_datasetname(row_id, field1, field2, ...).
        """
        for i, record in enumerate(self.records):
            args = [LogicNumber(value=i)]  # row_id
            
            for field_name in self.schema.keys():
                value = record[field_name]
                args.append(self._python_to_logic(value))
            
            yield LogicFact(
                head=LogicStruct(
                    functor=f"row_{self.dataset_name}",
                    args=args
                )
            )
    
    def _generate_field_facts(self) -> Iterable[LogicFact]:
        """
        Generate field facts: field_datasetname(row_id, field_name, field_value).
        """
        for i, record in enumerate(self.records):
            for field_name, value in record.items():
                if field_name in self.schema:
                    yield LogicFact(
                        head=LogicStruct(
                            functor=f"field_{self.dataset_name}",
                            args=[
                                LogicNumber(value=i),
                                LogicAtom(value=field_name),
                                self._python_to_logic(value),
                            ]
                        )
                    )
    
    def _python_to_logic(self, value: Any) -> LogicAtom | LogicNumber | LogicString:
        """Convert Python value to logic term."""
        if isinstance(value, bool):
            return LogicAtom(value="true" if value else "false")
        elif isinstance(value, (int, float)):
            return LogicNumber(value=value)
        elif isinstance(value, str):
            return LogicString(value=value)
        elif value is None:
            return LogicAtom(value="null")
        else:
            return LogicString(value=str(value))


# ============================================================================
# Model Registry Adapter
# ============================================================================

class ModelAdapter(LogicAdapter):
    """
    Adapter for exposing model registry as logic facts.
    
    Provides predicates like:
    - model(name, version, type)
    - model_metric(name, metric_name, metric_value)
    - model_param(name, param_name, param_value)
    
    This allows queries like:
    - "Find all classification models with accuracy > 0.9"
    - "Which model has the best F1 score?"
    """
    
    def __init__(self, models: List[Dict[str, Any]]):
        """
        Initialize model adapter.
        
        Args:
            models: List of model metadata dictionaries with keys:
                   name, version, type, metrics, params
        """
        self.models = models
    
    def get_predicates(self) -> List[str]:
        """Return predicates: model/3, model_metric/3, model_param/3."""
        return ["model", "model_metric", "model_param"]
    
    def get_arity(self, predicate: str) -> int:
        """Return predicate arity."""
        if predicate in ["model", "model_metric", "model_param"]:
            return 3
        else:
            raise ValueError(f"Unknown predicate: {predicate}")
    
    def get_facts(self, predicate: Optional[str] = None) -> Iterable[LogicFact]:
        """Generate facts for models."""
        if predicate is None or predicate == "model":
            yield from self._generate_model_facts()
        
        if predicate is None or predicate == "model_metric":
            yield from self._generate_metric_facts()
        
        if predicate is None or predicate == "model_param":
            yield from self._generate_param_facts()
    
    def _generate_model_facts(self) -> Iterable[LogicFact]:
        """Generate model facts: model(name, version, type)."""
        for model in self.models:
            yield LogicFact(
                head=LogicStruct(
                    functor="model",
                    args=[
                        LogicAtom(value=model.get("name", "unknown")),
                        LogicString(value=model.get("version", "1.0")),
                        LogicAtom(value=model.get("type", "unknown")),
                    ]
                )
            )
    
    def _generate_metric_facts(self) -> Iterable[LogicFact]:
        """Generate metric facts: model_metric(name, metric_name, value)."""
        for model in self.models:
            name = model.get("name", "unknown")
            metrics = model.get("metrics", {})
            
            for metric_name, metric_value in metrics.items():
                yield LogicFact(
                    head=LogicStruct(
                        functor="model_metric",
                        args=[
                            LogicAtom(value=name),
                            LogicAtom(value=metric_name),
                            LogicNumber(value=float(metric_value)),
                        ]
                    )
                )
    
    def _generate_param_facts(self) -> Iterable[LogicFact]:
        """Generate param facts: model_param(name, param_name, value)."""
        for model in self.models:
            name = model.get("name", "unknown")
            params = model.get("params", {})
            
            for param_name, param_value in params.items():
                yield LogicFact(
                    head=LogicStruct(
                        functor="model_param",
                        args=[
                            LogicAtom(value=name),
                            LogicAtom(value=param_name),
                            self._python_to_logic(param_value),
                        ]
                    )
                )
    
    def _python_to_logic(self, value: Any) -> LogicAtom | LogicNumber | LogicString:
        """Convert Python value to logic term."""
        if isinstance(value, bool):
            return LogicAtom(value="true" if value else "false")
        elif isinstance(value, (int, float)):
            return LogicNumber(value=value)
        elif isinstance(value, str):
            return LogicString(value=value)
        elif value is None:
            return LogicAtom(value="null")
        else:
            return LogicString(value=str(value))


# ============================================================================
# Adapter Registry
# ============================================================================

class AdapterRegistry:
    """
    Registry for managing logic adapters.
    
    Allows dynamic registration of adapters and provides unified access
    to facts from multiple data sources.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self.adapters: Dict[str, LogicAdapter] = {}
    
    def register(self, name: str, adapter: LogicAdapter) -> None:
        """Register an adapter with a name."""
        if name in self.adapters:
            raise ValueError(f"Adapter already registered: {name}")
        self.adapters[name] = adapter
    
    def get_adapter(self, name: str) -> Optional[LogicAdapter]:
        """Get an adapter by name."""
        return self.adapters.get(name)
    
    def get_all_facts(self) -> List[LogicFact]:
        """Get all facts from all adapters."""
        facts = []
        for adapter in self.adapters.values():
            facts.extend(adapter.get_facts())
        return facts
    
    def get_facts_for_predicate(self, predicate: str) -> List[LogicFact]:
        """Get facts for a specific predicate from all adapters."""
        facts = []
        for adapter in self.adapters.values():
            if predicate in adapter.get_predicates():
                facts.extend(adapter.get_facts(predicate))
        return facts


__all__ = [
    "LogicAdapter",
    "DatasetAdapter",
    "ModelAdapter",
    "AdapterRegistry",
]
