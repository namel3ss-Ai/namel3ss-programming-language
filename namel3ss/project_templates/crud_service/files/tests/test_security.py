"""
Unit tests for authentication and security utilities.

Tests JWT token validation, user extraction, and security helpers.
"""

from datetime import datetime, timedelta

import pytest
import jwt as pyjwt

from api.security import (
    User,
    TokenData,
    decode_jwt_token,
    token_data_to_user,
    create_access_token,
    verify_user_has_role,
    verify_user_has_any_role,
    verify_user_has_all_roles,
    TokenExpiredError,
    TokenInvalidError,
    TokenValidationError,
)
from config.settings import Settings


@pytest.fixture
def test_settings():
    """Create test settings with authentication config."""
    return Settings(
        database_url="postgresql://test:test@localhost:5432/test",
        jwt_secret_key="test-secret-key-at-least-32-chars-long-for-security",
        jwt_algorithm="HS256",
        jwt_access_token_expire_minutes=30,
        jwt_issuer="https://test.example.com",
        jwt_audience="test-app",
        auth_disabled=False,
    )


@pytest.fixture
def test_user():
    """Create test user."""
    return User(
        id="user_123",
        email="test@example.com",
        username="testuser",
        roles=["user", "admin"],
        is_active=True,
        tenant_id="tenant_abc",
        metadata={"department": "engineering"},
    )


class TestTokenValidation:
    """Test JWT token validation."""
    
    def test_decode_valid_token(self, test_settings):
        """Test decoding a valid JWT token."""
        # Create a valid token
        token = create_access_token(
            user_id="user_123",
            settings=test_settings,
            email="test@example.com",
            username="testuser",
            roles=["user", "admin"],
            tenant_id="tenant_abc",
        )
        
        # Decode and validate
        token_data = decode_jwt_token(token, test_settings)
        
        assert token_data.sub == "user_123"
        assert token_data.email == "test@example.com"
        assert token_data.username == "testuser"
        assert token_data.roles == ["user", "admin"]
        assert token_data.tenant_id == "tenant_abc"
        assert token_data.iss == test_settings.jwt_issuer
        assert token_data.aud == test_settings.jwt_audience
    
    def test_decode_expired_token(self, test_settings):
        """Test decoding an expired token."""
        # Create token that's already expired
        token = create_access_token(
            user_id="user_123",
            settings=test_settings,
            expires_delta=timedelta(seconds=-10),  # Expired 10 seconds ago
        )
        
        # Should raise TokenExpiredError
        with pytest.raises(TokenExpiredError, match="Token has expired"):
            decode_jwt_token(token, test_settings)
    
    def test_decode_invalid_signature(self, test_settings):
        """Test decoding a token with invalid signature."""
        # Create a token with different secret
        wrong_secret = "wrong-secret-key-different-from-settings-value"
        
        claims = {
            "sub": "user_123",
            "exp": datetime.utcnow() + timedelta(minutes=30),
            "iat": datetime.utcnow(),
        }
        
        token = pyjwt.encode(claims, wrong_secret, algorithm="HS256")
        
        # Should raise TokenInvalidError
        with pytest.raises(TokenInvalidError, match="Invalid token signature"):
            decode_jwt_token(token, test_settings)
    
    def test_decode_invalid_issuer(self, test_settings):
        """Test decoding a token with wrong issuer."""
        # Create token with wrong issuer
        claims = {
            "sub": "user_123",
            "exp": datetime.utcnow() + timedelta(minutes=30),
            "iat": datetime.utcnow(),
            "iss": "https://wrong-issuer.com",  # Wrong issuer
            "aud": test_settings.jwt_audience,
        }
        
        token = pyjwt.encode(claims, test_settings.jwt_secret_key, algorithm="HS256")
        
        # Should raise TokenValidationError
        with pytest.raises(TokenValidationError, match="Invalid token issuer"):
            decode_jwt_token(token, test_settings)
    
    def test_decode_invalid_audience(self, test_settings):
        """Test decoding a token with wrong audience."""
        # Create token with wrong audience
        claims = {
            "sub": "user_123",
            "exp": datetime.utcnow() + timedelta(minutes=30),
            "iat": datetime.utcnow(),
            "iss": test_settings.jwt_issuer,
            "aud": "wrong-audience",  # Wrong audience
        }
        
        token = pyjwt.encode(claims, test_settings.jwt_secret_key, algorithm="HS256")
        
        # Should raise TokenValidationError
        with pytest.raises(TokenValidationError, match="Invalid token audience"):
            decode_jwt_token(token, test_settings)
    
    def test_decode_missing_required_claims(self, test_settings):
        """Test decoding a token missing required claims."""
        # Create token without 'sub' claim
        claims = {
            "exp": datetime.utcnow() + timedelta(minutes=30),
            "iat": datetime.utcnow(),
        }
        
        token = pyjwt.encode(claims, test_settings.jwt_secret_key, algorithm="HS256")
        
        # Should raise TokenInvalidError
        with pytest.raises(TokenInvalidError):
            decode_jwt_token(token, test_settings)
    
    def test_decode_malformed_token(self, test_settings):
        """Test decoding a malformed token."""
        # Invalid JWT format
        token = "not.a.valid.jwt.token"
        
        # Should raise TokenInvalidError
        with pytest.raises(TokenInvalidError):
            decode_jwt_token(token, test_settings)
    
    def test_decode_without_issuer_validation(self):
        """Test decoding when issuer validation is not configured."""
        settings = Settings(
            database_url="postgresql://test:test@localhost:5432/test",
            jwt_secret_key="test-secret-key-at-least-32-chars-long",
            jwt_algorithm="HS256",
            jwt_issuer=None,  # No issuer validation
            jwt_audience=None,  # No audience validation
        )
        
        # Create token without issuer
        token = create_access_token(
            user_id="user_123",
            settings=settings,
        )
        
        # Should decode successfully
        token_data = decode_jwt_token(token, settings)
        assert token_data.sub == "user_123"


class TestUserConversion:
    """Test token data to user conversion."""
    
    def test_token_data_to_user(self):
        """Test converting token data to user model."""
        token_data = TokenData(
            sub="user_123",
            email="test@example.com",
            username="testuser",
            roles=["user", "admin"],
            tenant_id="tenant_abc",
        )
        
        user = token_data_to_user(token_data)
        
        assert user.id == "user_123"
        assert user.email == "test@example.com"
        assert user.username == "testuser"
        assert user.roles == ["user", "admin"]
        assert user.is_active is True
        assert user.tenant_id == "tenant_abc"
    
    def test_token_data_to_user_minimal(self):
        """Test converting minimal token data."""
        token_data = TokenData(sub="user_456")
        
        user = token_data_to_user(token_data)
        
        assert user.id == "user_456"
        assert user.email is None
        assert user.username is None
        assert user.roles == []
        assert user.is_active is True


class TestAccessTokenCreation:
    """Test JWT token creation."""
    
    def test_create_access_token_basic(self, test_settings):
        """Test creating a basic access token."""
        token = create_access_token(
            user_id="user_123",
            settings=test_settings,
        )
        
        # Decode to verify
        payload = pyjwt.decode(
            token,
            test_settings.jwt_secret_key,
            algorithms=[test_settings.jwt_algorithm],
            options={"verify_exp": False},  # Don't verify expiry for test
        )
        
        assert payload["sub"] == "user_123"
        assert "exp" in payload
        assert "iat" in payload
    
    def test_create_access_token_with_claims(self, test_settings):
        """Test creating token with additional claims."""
        token = create_access_token(
            user_id="user_123",
            settings=test_settings,
            email="test@example.com",
            username="testuser",
            roles=["user", "admin"],
            tenant_id="tenant_abc",
        )
        
        # Decode to verify
        payload = pyjwt.decode(
            token,
            test_settings.jwt_secret_key,
            algorithms=[test_settings.jwt_algorithm],
            audience=test_settings.jwt_audience,
            issuer=test_settings.jwt_issuer,
        )
        
        assert payload["sub"] == "user_123"
        assert payload["email"] == "test@example.com"
        assert payload["username"] == "testuser"
        assert payload["roles"] == ["user", "admin"]
        assert payload["tenant_id"] == "tenant_abc"
        assert payload["iss"] == test_settings.jwt_issuer
        assert payload["aud"] == test_settings.jwt_audience
    
    def test_create_access_token_custom_expiry(self, test_settings):
        """Test creating token with custom expiration."""
        token = create_access_token(
            user_id="user_123",
            settings=test_settings,
            expires_delta=timedelta(hours=2),
        )
        
        # Decode to verify
        payload = pyjwt.decode(
            token,
            test_settings.jwt_secret_key,
            algorithms=[test_settings.jwt_algorithm],
            options={"verify_exp": False},
        )
        
        exp_time = datetime.utcfromtimestamp(payload["exp"])
        iat_time = datetime.utcfromtimestamp(payload["iat"])
        
        # Should be approximately 2 hours
        delta = exp_time - iat_time
        assert 7190 <= delta.total_seconds() <= 7210  # 2 hours ± 10 seconds


class TestRoleVerification:
    """Test role-based access control helpers."""
    
    def test_verify_user_has_role(self, test_user):
        """Test checking if user has a specific role."""
        assert verify_user_has_role(test_user, "admin") is True
        assert verify_user_has_role(test_user, "user") is True
        assert verify_user_has_role(test_user, "superadmin") is False
    
    def test_verify_user_has_any_role(self, test_user):
        """Test checking if user has any of specified roles."""
        assert verify_user_has_any_role(test_user, ["admin", "superadmin"]) is True
        assert verify_user_has_any_role(test_user, ["user", "moderator"]) is True
        assert verify_user_has_any_role(test_user, ["superadmin", "moderator"]) is False
    
    def test_verify_user_has_all_roles(self, test_user):
        """Test checking if user has all specified roles."""
        assert verify_user_has_all_roles(test_user, ["admin", "user"]) is True
        assert verify_user_has_all_roles(test_user, ["admin"]) is True
        assert verify_user_has_all_roles(test_user, ["admin", "superadmin"]) is False
        assert verify_user_has_all_roles(test_user, ["superadmin", "moderator"]) is False
    
    def test_verify_empty_roles(self):
        """Test role verification with user having no roles."""
        user = User(id="user_123", roles=[])
        
        assert verify_user_has_role(user, "admin") is False
        assert verify_user_has_any_role(user, ["admin", "user"]) is False
        assert verify_user_has_all_roles(user, []) is True  # Empty list always True


class TestUserModel:
    """Test User model."""
    
    def test_user_model_creation(self):
        """Test creating a user model."""
        user = User(
            id="user_123",
            email="test@example.com",
            username="testuser",
            roles=["user"],
            is_active=True,
            tenant_id="tenant_abc",
            metadata={"key": "value"},
        )
        
        assert user.id == "user_123"
        assert user.email == "test@example.com"
        assert user.username == "testuser"
        assert user.roles == ["user"]
        assert user.is_active is True
        assert user.tenant_id == "tenant_abc"
        assert user.metadata == {"key": "value"}
    
    def test_user_model_defaults(self):
        """Test user model with default values."""
        user = User(id="user_123")
        
        assert user.id == "user_123"
        assert user.email is None
        assert user.username is None
        assert user.roles == []
        assert user.is_active is True
        assert user.tenant_id is None
        assert user.metadata == {}
