"""
Error handling for FastAPI application.

Provides consistent error responses and exception handlers.
"""

from typing import Any, Optional

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from models.schemas import ErrorResponse, ErrorDetail


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Handle Pydantic validation errors.
    
    Converts validation errors into consistent error responses.
    
    Args:
        request: FastAPI request
        exc: Validation error
        
    Returns:
        JSON error response
    """
    details = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"] if loc != "body")
        details.append(
            ErrorDetail(
                field=field or None,
                message=error["msg"],
                code=error["type"],
            )
        )
    
    error_response = ErrorResponse(
        error="validation_error",
        message="Request validation failed",
        details=details,
        request_id=request.headers.get("X-Request-ID"),
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump(),
    )


async def general_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Handle unexpected exceptions.
    
    Provides safe error responses without leaking implementation details.
    
    Args:
        request: FastAPI request
        exc: Exception
        
    Returns:
        JSON error response
    """
    # Log the full exception for debugging
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    error_response = ErrorResponse(
        error="internal_server_error",
        message="An unexpected error occurred",
        request_id=request.headers.get("X-Request-ID"),
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(),
    )


async def not_found_handler(
    request: Request,
    exc: Any,
) -> JSONResponse:
    """
    Handle 404 Not Found errors.
    
    Args:
        request: FastAPI request
        exc: Exception
        
    Returns:
        JSON error response
    """
    error_response = ErrorResponse(
        error="not_found",
        message="The requested resource was not found",
        request_id=request.headers.get("X-Request-ID"),
    )
    
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=error_response.model_dump(),
    )


def create_error_response(
    error_type: str,
    message: str,
    status_code: int = status.HTTP_400_BAD_REQUEST,
    details: Optional[list[ErrorDetail]] = None,
    request_id: Optional[str] = None,
) -> JSONResponse:
    """
    Create a standardized error response.
    
    Helper function for creating consistent error responses throughout the app.
    
    Args:
        error_type: Error type identifier (e.g., "validation_error")
        message: Human-readable error message
        status_code: HTTP status code
        details: Optional detailed error information
        request_id: Optional request ID for tracing
        
    Returns:
        JSON error response
    """
    error_response = ErrorResponse(
        error=error_type,
        message=message,
        details=details,
        request_id=request_id,
    )
    
    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump(),
    )
