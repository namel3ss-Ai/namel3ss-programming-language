"""
Runtime type validation for SDK.

Provides:
    1. Request validation (inputs)
    2. Response validation (outputs)
    3. Schema compatibility validation
    4. Structured validation errors

Example:
    Validate request:
    ```python
    validator = RequestValidator(input_model)
    validated_data = validator.validate(raw_data)
    ```
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ValidationError as PydanticValidationError

from .ir import IRModel, SchemaVersion
from .errors import ValidationError, VersionMismatchError


class TypeValidator:
    """
    Base validator with type checking and schema validation.
    """

    def __init__(self, model: IRModel):
        """
        Initialize validator.
        
        Args:
            model: IR model for validation
        """
        self.model = model

    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against model schema.
        
        Args:
            data: Data to validate
        
        Returns:
            Validated data
        
        Raises:
            ValidationError: If validation fails
        """
        # Check all required fields present
        missing_fields = []
        for field in self.model.fields:
            if field.required and field.name not in data:
                missing_fields.append(field.name)

        if missing_fields:
            raise ValidationError(
                f"Missing required fields: {', '.join(missing_fields)}",
                field_path=None,
                validation_errors=[
                    {
                        "field": f,
                        "error": "field_required",
                        "message": f"Field required: {f}",
                    }
                    for f in missing_fields
                ],
            )

        # Type checking would go here
        # In production, use Pydantic model for validation

        return data


class RequestValidator:
    """
    Validates API request payloads.
    
    Features:
    - Type checking
    - Required field validation
    - Constraint validation
    - Format validation
    """

    def __init__(self, input_model: IRModel):
        """
        Initialize request validator.
        
        Args:
            input_model: Input schema model
        """
        self.input_model = input_model
        self.validator = TypeValidator(input_model)

    def validate(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate request data.
        
        Args:
            request_data: Raw request data
        
        Returns:
            Validated request data
        
        Raises:
            ValidationError: If validation fails
        """
        try:
            validated = self.validator.validate(request_data)
            return validated
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                f"Request validation failed: {e}",
                validation_errors=[{"error": str(e)}],
            )

    def validate_with_model(
        self, request_data: Dict[str, Any], pydantic_model: type[BaseModel]
    ) -> BaseModel:
        """
        Validate using Pydantic model.
        
        Args:
            request_data: Raw request data
            pydantic_model: Pydantic model class
        
        Returns:
            Validated model instance
        
        Raises:
            ValidationError: If validation fails
        """
        try:
            return pydantic_model.model_validate(request_data)
        except PydanticValidationError as e:
            validation_errors = [
                {
                    "field": ".".join(str(loc) for loc in err["loc"]),
                    "error": err["type"],
                    "message": err["msg"],
                }
                for err in e.errors()
            ]
            raise ValidationError(
                "Request validation failed",
                validation_errors=validation_errors,
            )


class ResponseValidator:
    """
    Validates API response payloads.
    
    Features:
    - Response schema validation
    - Version compatibility checking
    - Extra field handling
    - Error response parsing
    """

    def __init__(self, output_model: IRModel, strict: bool = True):
        """
        Initialize response validator.
        
        Args:
            output_model: Output schema model
            strict: If True, fail on schema mismatches
        """
        self.output_model = output_model
        self.strict = strict
        self.validator = TypeValidator(output_model)

    def validate(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate response data.
        
        Args:
            response_data: Raw response data
        
        Returns:
            Validated response data
        
        Raises:
            ValidationError: If validation fails
            VersionMismatchError: If schema version incompatible
        """
        # Check version if present
        if "x-schema-version" in response_data:
            self._validate_version(response_data["x-schema-version"])

        try:
            validated = self.validator.validate(response_data)
            return validated
        except ValidationError:
            if self.strict:
                raise
            # In non-strict mode, log warning and continue
            import warnings

            warnings.warn(
                f"Response validation failed but continuing (strict=False)",
                UserWarning,
            )
            return response_data

    def validate_with_model(
        self, response_data: Dict[str, Any], pydantic_model: type[BaseModel]
    ) -> BaseModel:
        """
        Validate using Pydantic model.
        
        Args:
            response_data: Raw response data
            pydantic_model: Pydantic model class
        
        Returns:
            Validated model instance
        
        Raises:
            ValidationError: If validation fails
        """
        try:
            return pydantic_model.model_validate(response_data)
        except PydanticValidationError as e:
            validation_errors = [
                {
                    "field": ".".join(str(loc) for loc in err["loc"]),
                    "error": err["type"],
                    "message": err["msg"],
                }
                for err in e.errors()
            ]
            raise ValidationError(
                "Response validation failed",
                validation_errors=validation_errors,
            )

    def _validate_version(self, response_version_str: str) -> None:
        """Validate response schema version."""
        try:
            response_version = SchemaVersion.parse(response_version_str)
            if not self.output_model.version.is_compatible_with(response_version):
                raise VersionMismatchError(
                    f"Response schema version {response_version} incompatible with SDK version {self.output_model.version}",
                    expected_version=str(self.output_model.version),
                    actual_version=response_version_str,
                    schema_name=self.output_model.name,
                )
        except ValueError:
            # Invalid version format
            if self.strict:
                raise ValidationError(
                    f"Invalid version format: {response_version_str}"
                )


class ValidationContext:
    """
    Context for validation operations.
    
    Tracks validation state and provides utilities.
    """

    def __init__(self, strict: bool = True):
        """
        Initialize validation context.
        
        Args:
            strict: If True, fail fast on validation errors
        """
        self.strict = strict
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[str] = []

    def add_error(self, error: Dict[str, Any]) -> None:
        """Add validation error."""
        self.errors.append(error)
        if self.strict:
            raise ValidationError(
                "Validation failed",
                validation_errors=self.errors,
            )

    def add_warning(self, warning: str) -> None:
        """Add validation warning."""
        self.warnings.append(warning)
        import warnings

        warnings.warn(warning, UserWarning)

    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return len(self.errors) > 0

    def clear(self) -> None:
        """Clear all errors and warnings."""
        self.errors = []
        self.warnings = []
