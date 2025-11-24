"""
IR serialization - JSON import/export for intermediate representation.

Enables IR to be persisted, transmitted, and consumed by external tools
and runtime adapters.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Union

from .spec import BackendIR, FrontendIR


def serialize_backend_ir(ir: BackendIR) -> Dict[str, Any]:
    """
    Serialize BackendIR to JSON-compatible dictionary.
    
    Args:
        ir: Backend IR to serialize
        
    Returns:
        JSON-serializable dictionary
        
    Example:
        >>> ir = build_backend_ir(app)
        >>> data = serialize_backend_ir(ir)
        >>> json.dump(data, open("backend_ir.json", "w"))
    """
    return asdict(ir)


def deserialize_backend_ir(data: Dict[str, Any]) -> BackendIR:
    """
    Deserialize BackendIR from JSON-compatible dictionary.
    
    Args:
        data: JSON-deserialized dictionary
        
    Returns:
        BackendIR instance
        
    Example:
        >>> data = json.load(open("backend_ir.json"))
        >>> ir = deserialize_backend_ir(data)
    """
    return BackendIR(**data)


def serialize_frontend_ir(ir: FrontendIR) -> Dict[str, Any]:
    """
    Serialize FrontendIR to JSON-compatible dictionary.
    
    Args:
        ir: Frontend IR to serialize
        
    Returns:
        JSON-serializable dictionary
    """
    return asdict(ir)


def deserialize_frontend_ir(data: Dict[str, Any]) -> FrontendIR:
    """
    Deserialize FrontendIR from JSON-compatible dictionary.
    
    Args:
        data: JSON-deserialized dictionary
        
    Returns:
        FrontendIR instance
    """
    return FrontendIR(**data)


def write_backend_ir(ir: BackendIR, path: Union[str, Path]) -> None:
    """
    Write BackendIR to JSON file.
    
    Args:
        ir: Backend IR to write
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = serialize_backend_ir(ir)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def read_backend_ir(path: Union[str, Path]) -> BackendIR:
    """
    Read BackendIR from JSON file.
    
    Args:
        path: Input file path
        
    Returns:
        BackendIR instance
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return deserialize_backend_ir(data)


def write_frontend_ir(ir: FrontendIR, path: Union[str, Path]) -> None:
    """
    Write FrontendIR to JSON file.
    
    Args:
        ir: Frontend IR to write
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = serialize_frontend_ir(ir)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def read_frontend_ir(path: Union[str, Path]) -> FrontendIR:
    """
    Read FrontendIR from JSON file.
    
    Args:
        path: Input file path
        
    Returns:
        FrontendIR instance
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return deserialize_frontend_ir(data)


__all__ = [
    "serialize_backend_ir",
    "deserialize_backend_ir",
    "serialize_frontend_ir",
    "deserialize_frontend_ir",
    "write_backend_ir",
    "read_backend_ir",
    "write_frontend_ir",
    "read_frontend_ir",
]
