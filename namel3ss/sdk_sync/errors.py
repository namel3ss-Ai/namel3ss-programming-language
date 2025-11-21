"""
Error types for SDK Sync system.

Provides structured exceptions for all SDK generation and validation errors.
"""

from typing import Any, Dict, List, Optional


class SDKSyncError(Exception):
    """Base exception for all SDK Sync errors."""

    def __init__(
        self,
        message: str,
        code: str = "SDK001",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to structured dictionary."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "details": self.details,
        }


class SchemaRegistryError(SDKSyncError):
    """Error in schema registry operations."""

    def __init__(
        self,
        message: str,
        schema_name: Optional[str] = None,
        version: Optional[str] = None,
    ):
        super().__init__(
            message,
            code="SDK002",
            details={"schema_name": schema_name, "version": version},
        )
        self.schema_name = schema_name
        self.version = version


class CodegenError(SDKSyncError):
    """Error during code generation."""

    def __init__(
        self,
        message: str,
        target_language: str = "python",
        source_file: Optional[str] = None,
        line_number: Optional[int] = None,
    ):
        super().__init__(
            message,
            code="SDK003",
            details={
                "target_language": target_language,
                "source_file": source_file,
                "line_number": line_number,
            },
        )
        self.target_language = target_language
        self.source_file = source_file
        self.line_number = line_number


class VersionMismatchError(SDKSyncError):
    """Schema version incompatibility error."""

    def __init__(
        self,
        message: str,
        expected_version: str,
        actual_version: str,
        schema_name: Optional[str] = None,
    ):
        super().__init__(
            message,
            code="SDK004",
            details={
                "expected_version": expected_version,
                "actual_version": actual_version,
                "schema_name": schema_name,
            },
        )
        self.expected_version = expected_version
        self.actual_version = actual_version
        self.schema_name = schema_name


class ValidationError(SDKSyncError):
    """Runtime validation error."""

    def __init__(
        self,
        message: str,
        field_path: Optional[str] = None,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(
            message,
            code="SDK005",
            details={
                "field_path": field_path,
                "validation_errors": validation_errors or [],
            },
        )
        self.field_path = field_path
        self.validation_errors = validation_errors or []


class MigrationError(SDKSyncError):
    """Schema migration error."""

    def __init__(
        self,
        message: str,
        from_version: str,
        to_version: str,
        schema_name: Optional[str] = None,
    ):
        super().__init__(
            message,
            code="SDK006",
            details={
                "from_version": from_version,
                "to_version": to_version,
                "schema_name": schema_name,
            },
        )
        self.from_version = from_version
        self.to_version = to_version
        self.schema_name = schema_name


class ExportError(SDKSyncError):
    """Error exporting schemas from N3 runtime."""

    def __init__(self, message: str, backend_url: Optional[str] = None):
        super().__init__(
            message,
            code="SDK007",
            details={"backend_url": backend_url},
        )
        self.backend_url = backend_url


class ImportError(SDKSyncError):
    """Error importing schemas into registry."""

    def __init__(self, message: str, source_path: Optional[str] = None):
        super().__init__(
            message,
            code="SDK008",
            details={"source_path": source_path},
        )
        self.source_path = source_path
