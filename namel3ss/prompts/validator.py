"""
Output validation for structured prompts.

Validates LLM-generated outputs against declared output schemas,
checking types, enums, required fields, and structure.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from namel3ss.ast import OutputSchema, OutputField, OutputFieldType
from namel3ss.errors import N3Error


class ValidationError(N3Error):
    """Raised when output validation fails."""
    
    def __init__(self, message: str, field_path: Optional[str] = None, value: Any = None):
        super().__init__(message)
        self.field_path = field_path
        self.value = value


@dataclass
class ValidationResult:
    """Result of output validation."""
    
    valid: bool
    errors: List[ValidationError]
    validated_output: Optional[Dict[str, Any]] = None
    
    def raise_if_invalid(self) -> None:
        """Raise the first validation error if validation failed."""
        if not self.valid and self.errors:
            raise self.errors[0]


class OutputValidator:
    """
    Validates LLM outputs against OutputSchema definitions.
    
    Example:
        validator = OutputValidator(output_schema)
        result = validator.validate(llm_output)
        if result.valid:
            use(result.validated_output)
        else:
            handle_errors(result.errors)
    """
    
    def __init__(self, schema: OutputSchema):
        """
        Initialize validator with an output schema.
        
        Args:
            schema: OutputSchema AST node
        """
        self.schema = schema
    
    def validate(self, output: Union[str, Dict[str, Any]]) -> ValidationResult:
        """
        Validate LLM output against the schema.
        
        Args:
            output: LLM output (JSON string or parsed dict)
            
        Returns:
            ValidationResult with errors and validated output
        """
        errors: List[ValidationError] = []
        
        # Parse JSON if string
        if isinstance(output, str):
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError as e:
                return ValidationResult(
                    valid=False,
                    errors=[ValidationError(f"Invalid JSON: {e}", field_path=None, value=output)]
                )
        else:
            parsed = output
        
        # Must be a dict
        if not isinstance(parsed, dict):
            return ValidationResult(
                valid=False,
                errors=[ValidationError(f"Expected object, got {type(parsed).__name__}", value=parsed)]
            )
        
        validated: Dict[str, Any] = {}
        
        # Validate each field
        for field in self.schema.fields:
            field_errors = self._validate_field(field, parsed, validated)
            errors.extend(field_errors)
        
        # Check for unexpected fields (strict validation)
        expected_fields = {f.name for f in self.schema.fields}
        unexpected = set(parsed.keys()) - expected_fields
        if unexpected:
            for unexpected_field in unexpected:
                errors.append(ValidationError(
                    f"Unexpected field: {unexpected_field}",
                    field_path=unexpected_field,
                    value=parsed[unexpected_field]
                ))
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            validated_output=validated if len(errors) == 0 else None
        )
    
    def _validate_field(
        self,
        field: OutputField,
        parsed: Dict[str, Any],
        validated: Dict[str, Any]
    ) -> List[ValidationError]:
        """
        Validate a single field.
        
        Args:
            field: Field definition
            parsed: Parsed output dict
            validated: Dict to populate with validated values
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors: List[ValidationError] = []
        field_name = field.name
        
        # Check if field is present
        if field_name not in parsed:
            if field.required:
                errors.append(ValidationError(
                    f"Missing required field: {field_name}",
                    field_path=field_name
                ))
            return errors
        
        value = parsed[field_name]
        
        # Validate field type
        type_errors = self._validate_field_type(
            field_name,
            value,
            field.field_type
        )
        errors.extend(type_errors)
        
        # If validation passed, add to validated dict
        if not type_errors:
            validated[field_name] = value
        
        return errors
    
    def _validate_field_type(
        self,
        field_path: str,
        value: Any,
        field_type: OutputFieldType
    ) -> List[ValidationError]:
        """
        Validate a value against a field type.
        
        Args:
            field_path: Dot-separated field path for error reporting
            value: Value to validate
            field_type: Expected type
            
        Returns:
            List of validation errors
        """
        errors: List[ValidationError] = []
        
        # Handle nullable
        if value is None:
            if not field_type.nullable:
                errors.append(ValidationError(
                    f"Field '{field_path}' cannot be null",
                    field_path=field_path,
                    value=value
                ))
            return errors
        
        # Validate based on base type
        if field_type.base_type == 'string':
            if not isinstance(value, str):
                errors.append(ValidationError(
                    f"Field '{field_path}' must be a string, got {type(value).__name__}",
                    field_path=field_path,
                    value=value
                ))
        
        elif field_type.base_type == 'int':
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(ValidationError(
                    f"Field '{field_path}' must be an integer, got {type(value).__name__}",
                    field_path=field_path,
                    value=value
                ))
        
        elif field_type.base_type == 'float':
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(ValidationError(
                    f"Field '{field_path}' must be a number, got {type(value).__name__}",
                    field_path=field_path,
                    value=value
                ))
        
        elif field_type.base_type == 'bool':
            if not isinstance(value, bool):
                errors.append(ValidationError(
                    f"Field '{field_path}' must be a boolean, got {type(value).__name__}",
                    field_path=field_path,
                    value=value
                ))
        
        elif field_type.base_type == 'enum':
            # Enum validation
            if not field_type.enum_values:
                errors.append(ValidationError(
                    f"Field '{field_path}' has enum type with no values (schema error)",
                    field_path=field_path
                ))
            elif not isinstance(value, str):
                errors.append(ValidationError(
                    f"Field '{field_path}' enum value must be a string, got {type(value).__name__}",
                    field_path=field_path,
                    value=value
                ))
            elif value not in field_type.enum_values:
                allowed = ', '.join(f'"{v}"' for v in field_type.enum_values)
                errors.append(ValidationError(
                    f"Field '{field_path}' has invalid enum value '{value}'. Allowed: {allowed}",
                    field_path=field_path,
                    value=value
                ))
        
        elif field_type.base_type == 'list':
            if not isinstance(value, list):
                errors.append(ValidationError(
                    f"Field '{field_path}' must be a list, got {type(value).__name__}",
                    field_path=field_path,
                    value=value
                ))
            elif field_type.element_type:
                # Validate each element
                for idx, elem in enumerate(value):
                    elem_path = f"{field_path}[{idx}]"
                    elem_errors = self._validate_field_type(
                        elem_path,
                        elem,
                        field_type.element_type
                    )
                    errors.extend(elem_errors)
        
        elif field_type.base_type == 'object':
            if not isinstance(value, dict):
                errors.append(ValidationError(
                    f"Field '{field_path}' must be an object, got {type(value).__name__}",
                    field_path=field_path,
                    value=value
                ))
            elif field_type.nested_fields:
                # Validate nested object fields
                for nested_field in field_type.nested_fields:
                    nested_path = f"{field_path}.{nested_field.name}"
                    
                    if nested_field.name not in value:
                        if nested_field.required:
                            errors.append(ValidationError(
                                f"Missing required nested field: {nested_path}",
                                field_path=nested_path
                            ))
                    else:
                        nested_value = value[nested_field.name]
                        nested_errors = self._validate_field_type(
                            nested_path,
                            nested_value,
                            nested_field.field_type
                        )
                        errors.extend(nested_errors)
        
        return errors
    
    def validate_and_raise(self, output: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate output and raise ValidationError if invalid.
        
        Args:
            output: LLM output to validate
            
        Returns:
            Validated output dict
            
        Raises:
            ValidationError: If validation fails
        """
        result = self.validate(output)
        if not result.valid:
            # Raise first error with all error details
            if result.errors:
                first_error = result.errors[0]
                all_errors = "; ".join(str(e) for e in result.errors)
                raise ValidationError(
                    f"Output validation failed: {all_errors}",
                    field_path=first_error.field_path,
                    value=first_error.value
                )
            else:
                raise ValidationError("Output validation failed with unknown error")
        
        return result.validated_output  # type: ignore


def validate_output(
    output: Union[str, Dict[str, Any]],
    schema: OutputSchema
) -> ValidationResult:
    """
    Convenience function to validate output against a schema.
    
    Args:
        output: LLM output (JSON string or dict)
        schema: OutputSchema to validate against
        
    Returns:
        ValidationResult
    """
    validator = OutputValidator(schema)
    return validator.validate(output)


__all__ = [
    "OutputValidator",
    "ValidationError",
    "ValidationResult",
    "validate_output",
]
