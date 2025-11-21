"""
Main FastAPI application for {{ project_name }}.

Provides HTTP API with OpenAPI docs, CORS, logging, and lifecycle management.
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from config.settings import get_settings
from api import (
    router,
    init_db_pool,
    close_db_pool,
    validation_exception_handler,
    general_exception_handler,
)


# Configure logging
def setup_logging() -> None:
    """Configure application logging."""
    settings = get_settings()
    
    log_level = getattr(logging, settings.log_level)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for resource initialization and cleanup.
    
    Args:
        app: FastAPI application instance
        
    Yields:
        None during application runtime
    """
    settings = get_settings()
    logger = logging.getLogger(__name__)
    
    # Startup
    logger.info(f"Starting {settings.app_name} in {settings.environment} mode")
    
    try:
        # Initialize database connection pool
        await init_db_pool(settings)
        logger.info("Database connection pool initialized")
        
        # Additional startup tasks can go here
        # Example: Initialize cache, load ML models, etc.
        
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    
    logger.info(f"{settings.app_name} started successfully")
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {settings.app_name}")
    
    try:
        # Close database connection pool
        await close_db_pool()
        logger.info("Database connection pool closed")
        
        # Additional cleanup tasks can go here
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}", exc_info=True)
    
    logger.info(f"{settings.app_name} shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()
    setup_logging()
    
    app = FastAPI(
        title=settings.app_name,
        description=f"RESTful CRUD API for {settings.app_name}",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        debug=settings.debug,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Total-Count"],
    )
    
    # Add GZip compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """Add unique request ID to each request."""
        import uuid
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    
    # Add logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests and responses."""
        logger = logging.getLogger(__name__)
        request_id = getattr(request.state, "request_id", "unknown")
        
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"[{request_id}] from {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            response = await call_next(request)
            logger.info(
                f"Response: {request.method} {request.url.path} "
                f"[{request_id}] status={response.status_code}"
            )
            return response
        except Exception as e:
            logger.error(
                f"Error: {request.method} {request.url.path} "
                f"[{request_id}] error={str(e)}",
                exc_info=True
            )
            raise
    
    # Register exception handlers
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    
    # Include API router
    app.include_router(router)
    
    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """
        Health check endpoint.
        
        Returns basic health status. Can be extended to check database
        connectivity, cache status, etc.
        """
        return {
            "status": "healthy",
            "environment": settings.environment,
            "version": "1.0.0",
        }
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """
        Root endpoint with API information.
        """
        return {
            "name": settings.app_name,
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    """
    Run application with uvicorn.
    
    For production, use:
        uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
    """
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
