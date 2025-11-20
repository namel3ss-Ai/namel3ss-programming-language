"""
Tests for authentication system.

Tests user registration, login, token management, and permissions.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from n3_server.main import app
from n3_server.db.models import Base
from n3_server.auth.models import User, ProjectMember, Role
from n3_server.db.models import Project
from n3_server.api.auth import get_db
from n3_server.auth.security import get_password_hash

# In-memory database for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def test_db():
    """Create test database."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session() as session:
        yield session
    
    await engine.dispose()


@pytest.fixture
def override_get_db(test_db: AsyncSession):
    """Override database dependency."""
    async def _get_db():
        yield test_db
    
    app.dependency_overrides[get_db] = _get_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def test_user(test_db: AsyncSession):
    """Create test user."""
    from nanoid import generate
    
    user = User(
        id=generate(size=12),
        email="test@example.com",
        username="testuser",
        hashed_password=get_password_hash("password123"),
        full_name="Test User",
        is_active=True,
        is_superuser=False,
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    return user


@pytest.fixture
async def auth_headers(test_user: User):
    """Create authorization headers for test user."""
    from n3_server.auth.security import create_access_token
    
    token = create_access_token(data={"sub": test_user.id})
    return {"Authorization": f"Bearer {token}"}


class TestUserRegistration:
    """Test user registration."""
    
    @pytest.mark.asyncio
    async def test_register_success(self, override_get_db):
        """Test successful user registration."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/auth/register",
                json={
                    "email": "new@example.com",
                    "username": "newuser",
                    "password": "password123",
                    "full_name": "New User",
                },
            )
        
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "new@example.com"
        assert data["username"] == "newuser"
        assert data["full_name"] == "New User"
        assert "id" in data
        assert "hashed_password" not in data
    
    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, test_user, override_get_db):
        """Test registration with duplicate email."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/auth/register",
                json={
                    "email": test_user.email,
                    "username": "different",
                    "password": "password123",
                },
            )
        
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_register_duplicate_username(self, test_user, override_get_db):
        """Test registration with duplicate username."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/auth/register",
                json={
                    "email": "different@example.com",
                    "username": test_user.username,
                    "password": "password123",
                },
            )
        
        assert response.status_code == 400
        assert "already taken" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_register_weak_password(self, override_get_db):
        """Test registration with weak password."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/auth/register",
                json={
                    "email": "new@example.com",
                    "username": "newuser",
                    "password": "123",  # Too short
                },
            )
        
        assert response.status_code == 422  # Validation error


class TestUserLogin:
    """Test user login."""
    
    @pytest.mark.asyncio
    async def test_login_with_username(self, test_user, override_get_db):
        """Test login with username."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/auth/login",
                json={
                    "username": test_user.username,
                    "password": "password123",
                },
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_login_with_email(self, test_user, override_get_db):
        """Test login with email."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/auth/login",
                json={
                    "username": test_user.email,  # Can use email as username
                    "password": "password123",
                },
            )
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_login_wrong_password(self, test_user, override_get_db):
        """Test login with wrong password."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/auth/login",
                json={
                    "username": test_user.username,
                    "password": "wrongpassword",
                },
            )
        
        assert response.status_code == 401
        assert "Incorrect" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_login_nonexistent_user(self, override_get_db):
        """Test login with nonexistent user."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/auth/login",
                json={
                    "username": "nonexistent",
                    "password": "password123",
                },
            )
        
        assert response.status_code == 401


class TestTokenManagement:
    """Test token management."""
    
    @pytest.mark.asyncio
    async def test_refresh_token(self, test_user, override_get_db):
        """Test refreshing access token."""
        from n3_server.auth.security import create_refresh_token
        
        refresh_token = create_refresh_token(data={"sub": test_user.id})
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/auth/refresh",
                json={"refresh_token": refresh_token},
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
    
    @pytest.mark.asyncio
    async def test_refresh_with_access_token(self, test_user, override_get_db):
        """Test that access token cannot be used for refresh."""
        from n3_server.auth.security import create_access_token
        
        access_token = create_access_token(data={"sub": test_user.id})
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/auth/refresh",
                json={"refresh_token": access_token},
            )
        
        assert response.status_code == 401


class TestUserProfile:
    """Test user profile management."""
    
    @pytest.mark.asyncio
    async def test_get_current_user(self, test_user, auth_headers, override_get_db):
        """Test getting current user info."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(
                "/api/auth/me",
                headers=auth_headers,
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_user.id
        assert data["email"] == test_user.email
        assert data["username"] == test_user.username
    
    @pytest.mark.asyncio
    async def test_get_current_user_without_auth(self, override_get_db):
        """Test getting current user without authentication."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/auth/me")
        
        assert response.status_code == 403  # No authorization header
    
    @pytest.mark.asyncio
    async def test_update_user_profile(self, test_user, auth_headers, override_get_db):
        """Test updating user profile."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.patch(
                "/api/auth/me",
                headers=auth_headers,
                json={
                    "full_name": "Updated Name",
                    "email": "updated@example.com",
                },
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["full_name"] == "Updated Name"
        assert data["email"] == "updated@example.com"
    
    @pytest.mark.asyncio
    async def test_change_password(self, test_user, auth_headers, override_get_db):
        """Test changing password."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/auth/change-password",
                headers=auth_headers,
                json={
                    "current_password": "password123",
                    "new_password": "newpassword123",
                },
            )
        
        assert response.status_code == 204
        
        # Test login with new password
        response = await client.post(
            "/api/auth/login",
            json={
                "username": test_user.username,
                "password": "newpassword123",
            },
        )
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_change_password_wrong_current(
        self,
        test_user,
        auth_headers,
        override_get_db,
    ):
        """Test changing password with wrong current password."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/auth/change-password",
                headers=auth_headers,
                json={
                    "current_password": "wrongpassword",
                    "new_password": "newpassword123",
                },
            )
        
        assert response.status_code == 400


class TestProjectPermissions:
    """Test project permission system."""
    
    @pytest.fixture
    async def test_project(self, test_user, test_db):
        """Create test project."""
        from nanoid import generate
        
        project = Project(
            id=generate(size=12),
            name="Test Project",
            owner_id=test_user.id,
            graph_data={},
        )
        test_db.add(project)
        await test_db.commit()
        await test_db.refresh(project)
        return project
    
    @pytest.fixture
    async def other_user(self, test_db):
        """Create another test user."""
        from nanoid import generate
        
        user = User(
            id=generate(size=12),
            email="other@example.com",
            username="otheruser",
            hashed_password=get_password_hash("password123"),
            is_active=True,
        )
        test_db.add(user)
        await test_db.commit()
        await test_db.refresh(user)
        return user
    
    @pytest.mark.asyncio
    async def test_add_project_member(
        self,
        test_project,
        other_user,
        auth_headers,
        override_get_db,
    ):
        """Test adding a member to a project."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                f"/api/projects/{test_project.id}/members",
                headers=auth_headers,
                json={
                    "email": other_user.email,
                    "role": "editor",
                },
            )
        
        assert response.status_code == 201
        data = response.json()
        assert data["user_email"] == other_user.email
        assert data["role"] == "editor"
    
    @pytest.mark.asyncio
    async def test_list_project_members(
        self,
        test_project,
        other_user,
        test_db,
        auth_headers,
        override_get_db,
    ):
        """Test listing project members."""
        from nanoid import generate
        
        # Add member
        membership = ProjectMember(
            id=generate(size=12),
            project_id=test_project.id,
            user_id=other_user.id,
            role=Role.VIEWER,
        )
        test_db.add(membership)
        await test_db.commit()
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(
                f"/api/projects/{test_project.id}/members",
                headers=auth_headers,
            )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["user_email"] == other_user.email


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
