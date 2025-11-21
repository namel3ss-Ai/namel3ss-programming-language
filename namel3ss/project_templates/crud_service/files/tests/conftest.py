"""
Pytest configuration and fixtures.

Provides shared fixtures for testing the CRUD service.
"""

import asyncio
import os
from typing import AsyncGenerator, Generator
from uuid import uuid4

import asyncpg
import pytest
from httpx import AsyncClient

# Set test environment before importing app
os.environ["ENVIRONMENT"] = "development"
os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/{{ project_name | lower | replace('-', '_') }}_test"
)
os.environ["DEBUG"] = "true"
os.environ["LOG_LEVEL"] = "WARNING"

from main import app
from config.settings import get_settings, reset_settings
from api.dependencies import init_db_pool, close_db_pool, get_db_pool
from models.domain import {{ entity_name }}
from repository import Postgres{{ entity_name }}Repository


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def db_pool() -> AsyncGenerator[asyncpg.Pool, None]:
    """Create database connection pool for tests."""
    settings = get_settings()
    pool = await init_db_pool(settings)
    
    yield pool
    
    await close_db_pool()


@pytest.fixture
async def db_connection(db_pool: asyncpg.Pool) -> AsyncGenerator[asyncpg.Connection, None]:
    """Provide a database connection with transaction rollback."""
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            yield conn
            # Transaction automatically rolls back after test


@pytest.fixture
async def repository(db_pool: asyncpg.Pool) -> Postgres{{ entity_name }}Repository:
    """Provide repository instance."""
    return Postgres{{ entity_name }}Repository(db_pool)


@pytest.fixture
async def clean_database(db_pool: asyncpg.Pool) -> AsyncGenerator[None, None]:
    """Clean database before and after test."""
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM {{ table_name }}")
    
    yield
    
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM {{ table_name }}")


@pytest.fixture
def sample_item() -> {{ entity_name }}:
    """Create a sample item for testing."""
    return {{ entity_name }}(
        name="Test Item",
        description="A test item for unit testing",
        quantity=50,
        price=29.99,
        is_active=True,
        tags=["test", "sample"],
        metadata={"color": "blue", "size": "medium"},
    )


@pytest.fixture
async def created_item(
    repository: Postgres{{ entity_name }}Repository,
    sample_item: {{ entity_name }},
    clean_database,
) -> {{ entity_name }}:
    """Create and persist a sample item."""
    return await repository.create(sample_item)


@pytest.fixture
async def multiple_items(
    repository: Postgres{{ entity_name }}Repository,
    clean_database,
) -> list[{{ entity_name }}]:
    """Create multiple test items."""
    items = []
    for i in range(5):
        item = {{ entity_name }}(
            name=f"Item {i}",
            description=f"Description {i}",
            quantity=i * 10,
            price=float(i * 10 + 9.99),
            is_active=i % 2 == 0,
            tags=[f"tag{i}", "common"],
            metadata={"index": i},
        )
        created = await repository.create(item)
        items.append(created)
    
    return items


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Provide HTTP client for API testing."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def tenant_id() -> str:
    """Provide a test tenant ID."""
    return f"tenant_{uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
def reset_config():
    """Reset configuration after each test."""
    yield
    reset_settings()
