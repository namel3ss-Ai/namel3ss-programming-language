"""
Conformance test runner for Namel3ss Language 1.0.

This module provides the core infrastructure for running conformance tests
that validate language-level behavior. The conformance suite is designed to be
implementation-agnostic and can be used by any Namel3ss implementation to verify
correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
import yaml


class TestPhase(str, Enum):
    """Phases of compilation/execution that can be tested."""
    PARSE = "parse"
    RESOLVE = "resolve"
    TYPECHECK = "typecheck"
    CODEGEN = "codegen"
    RUNTIME = "runtime"


class TestStatus(str, Enum):
    """Expected outcome status for a test phase."""
    SUCCESS = "success"
    ERROR = "error"


class DiagnosticSeverity(str, Enum):
    """Severity level for diagnostics."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class SourceLocation:
    """Location information for errors and diagnostics."""
    file: str
    line: int
    column: int
    length: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "file": self.file,
            "line": self.line,
            "column": self.column,
        }
        if self.length is not None:
            result["length"] = self.length
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SourceLocation:
        return cls(
            file=data["file"],
            line=data["line"],
            column=data["column"],
            length=data.get("length")
        )


@dataclass
class Diagnostic:
    """Expected diagnostic (error/warning) from compilation."""
    severity: DiagnosticSeverity
    code: str
    message: str
    location: Optional[SourceLocation] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
        }
        if self.location:
            result["location"] = self.location.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Diagnostic:
        return cls(
            severity=DiagnosticSeverity(data["severity"]),
            code=data["code"],
            message=data["message"],
            location=SourceLocation.from_dict(data["location"]) if "location" in data else None
        )


@dataclass
class ParseExpectation:
    """Expected outcome of parse phase."""
    status: TestStatus
    ast: Optional[Dict[str, Any]] = None
    errors: List[Diagnostic] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ParseExpectation:
        return cls(
            status=TestStatus(data["status"]),
            ast=data.get("ast"),
            errors=[Diagnostic.from_dict(e) for e in data.get("errors", [])]
        )


@dataclass
class TypecheckExpectation:
    """Expected outcome of typecheck phase."""
    status: TestStatus
    diagnostics: List[Diagnostic] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TypecheckExpectation:
        return cls(
            status=TestStatus(data["status"]),
            diagnostics=[Diagnostic.from_dict(d) for d in data.get("diagnostics", [])]
        )


@dataclass
class RuntimeError:
    """Expected runtime error."""
    type: str
    message: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RuntimeError:
        return cls(
            type=data["type"],
            message=data["message"]
        )


@dataclass
class RuntimeExpectation:
    """Expected outcome of runtime phase."""
    status: TestStatus
    timeout_ms: int = 5000
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    exit_code: Optional[int] = None
    error: Optional[RuntimeError] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RuntimeExpectation:
        return cls(
            status=TestStatus(data["status"]),
            timeout_ms=data.get("timeout_ms", 5000),
            stdout=data.get("stdout"),
            stderr=data.get("stderr"),
            exit_code=data.get("exit_code"),
            error=RuntimeError.from_dict(data["error"]) if "error" in data else None
        )


@dataclass
class SourceFile:
    """Source file specification."""
    path: Optional[str] = None
    content: Optional[str] = None
    
    def get_content(self, base_path: Path) -> str:
        """Get source content, either from file or inline."""
        if self.content is not None:
            return self.content
        if self.path is not None:
            file_path = base_path / self.path
            return file_path.read_text(encoding="utf-8")
        raise ValueError("Source must have either path or content")
    
    @classmethod
    def from_dict(cls, data: Union[str, Dict[str, Any]]) -> SourceFile:
        if isinstance(data, str):
            return cls(path=data)
        return cls(
            path=data.get("path"),
            content=data.get("content")
        )


@dataclass
class TestExpectations:
    """All expected outcomes for a test."""
    parse: Optional[ParseExpectation] = None
    typecheck: Optional[TypecheckExpectation] = None
    runtime: Optional[RuntimeExpectation] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TestExpectations:
        return cls(
            parse=ParseExpectation.from_dict(data["parse"]) if "parse" in data else None,
            typecheck=TypecheckExpectation.from_dict(data["typecheck"]) if "typecheck" in data else None,
            runtime=RuntimeExpectation.from_dict(data["runtime"]) if "runtime" in data else None
        )


@dataclass
class TestConfig:
    """Optional test-specific configuration."""
    strict_ast_match: bool = False
    allow_extra_diagnostics: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TestConfig:
        return cls(
            strict_ast_match=data.get("strict_ast_match", False),
            allow_extra_diagnostics=data.get("allow_extra_diagnostics", False)
        )


@dataclass
class ConformanceTestDescriptor:
    """
    Complete conformance test descriptor.
    
    This is the machine-readable representation of a conformance test that
    external implementations can consume to validate their behavior.
    """
    
    # Required metadata
    spec_version: str
    language_version: str
    test_id: str
    category: Literal["parse", "types", "runtime"]
    name: str
    
    # Test phases
    phases: List[TestPhase]
    
    # Source files
    sources: List[SourceFile]
    
    # Expected outcomes
    expect: TestExpectations
    
    # Optional fields
    description: Optional[str] = None
    config: TestConfig = field(default_factory=TestConfig)
    optional: bool = False
    tags: List[str] = field(default_factory=list)
    
    # File metadata
    _file_path: Optional[Path] = field(default=None, repr=False)
    
    @classmethod
    def from_file(cls, path: Path) -> ConformanceTestDescriptor:
        """Load test descriptor from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        descriptor = cls.from_dict(data)
        descriptor._file_path = path
        return descriptor
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConformanceTestDescriptor:
        """Create descriptor from dictionary."""
        return cls(
            spec_version=data["spec_version"],
            language_version=data["language_version"],
            test_id=data["test_id"],
            category=data["category"],
            name=data["name"],
            phases=[TestPhase(p) for p in data["phases"]],
            sources=[SourceFile.from_dict(s) for s in data["sources"]],
            expect=TestExpectations.from_dict(data["expect"]),
            description=data.get("description"),
            config=TestConfig.from_dict(data.get("config", {})),
            optional=data.get("optional", False),
            tags=data.get("tags", [])
        )
    
    def get_base_path(self) -> Path:
        """Get base path for resolving relative source paths."""
        if self._file_path:
            return self._file_path.parent
        return Path.cwd()
    
    def validate(self) -> List[str]:
        """
        Validate descriptor structure and contents.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Version format validation
        if not self._is_valid_version(self.spec_version):
            errors.append(f"Invalid spec_version format: {self.spec_version}")
        if not self._is_valid_version(self.language_version):
            errors.append(f"Invalid language_version format: {self.language_version}")
        
        # Test ID format
        if not self.test_id or not self.test_id.strip():
            errors.append("test_id cannot be empty")
        
        # Category validation
        if self.category not in ["parse", "types", "runtime"]:
            errors.append(f"Invalid category: {self.category}")
        
        # Phases validation
        if not self.phases:
            errors.append("At least one phase must be specified")
        
        # Sources validation
        if not self.sources:
            errors.append("At least one source must be specified")
        
        for i, source in enumerate(self.sources):
            if source.path is None and source.content is None:
                errors.append(f"Source {i}: must have either path or content")
            if source.path is not None and source.content is not None:
                errors.append(f"Source {i}: cannot have both path and content")
        
        # Expectations validation
        for phase in self.phases:
            phase_name = phase.value
            expectation = getattr(self.expect, phase_name, None)
            if expectation is None:
                errors.append(f"Missing expectation for phase: {phase_name}")
        
        return errors
    
    @staticmethod
    def _is_valid_version(version: str) -> bool:
        """Check if version string is valid semver."""
        parts = version.split(".")
        if len(parts) != 3:
            return False
        try:
            for part in parts:
                int(part)
            return True
        except ValueError:
            return False


def discover_conformance_tests(
    root_dir: Path,
    category: Optional[str] = None,
    test_id: Optional[str] = None
) -> List[ConformanceTestDescriptor]:
    """
    Discover all conformance test descriptors in a directory.
    
    Args:
        root_dir: Root directory to search (e.g., tests/conformance/v1/)
        category: Optional category filter (parse, types, runtime)
        test_id: Optional specific test ID to load
    
    Returns:
        List of valid conformance test descriptors
    """
    descriptors = []
    
    # Find all .test.yaml files
    pattern = "**/*.test.yaml"
    for test_file in root_dir.rglob(pattern):
        try:
            descriptor = ConformanceTestDescriptor.from_file(test_file)
            
            # Apply filters
            if category and descriptor.category != category:
                continue
            if test_id and descriptor.test_id != test_id:
                continue
            
            # Validate descriptor
            errors = descriptor.validate()
            if errors:
                print(f"Warning: Invalid test descriptor {test_file}:")
                for error in errors:
                    print(f"  - {error}")
                continue
            
            descriptors.append(descriptor)
        except Exception as e:
            print(f"Error loading test {test_file}: {e}")
            continue
    
    return sorted(descriptors, key=lambda d: d.test_id)


__all__ = [
    "TestPhase",
    "TestStatus",
    "DiagnosticSeverity",
    "SourceLocation",
    "Diagnostic",
    "ParseExpectation",
    "TypecheckExpectation",
    "RuntimeExpectation",
    "RuntimeError",
    "SourceFile",
    "TestExpectations",
    "TestConfig",
    "ConformanceTestDescriptor",
    "discover_conformance_tests",
]
