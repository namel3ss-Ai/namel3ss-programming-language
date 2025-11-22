# Authentication Guide

## Overview

This service implements **production-ready JWT-based authentication** using industry-standard practices. All write operations (create, update, delete) require authentication, while read operations are public by default.

## Quick Start

### 1. Configure Authentication

Set the following environment variables:

```bash
# Required - JWT Secret Key (minimum 32 characters)
JWT_SECRET_KEY="your-cryptographically-secure-random-key-here"

# Optional - JWT Configuration
JWT_ALGORITHM="HS256"                    # HS256 (default) or RS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30       # Token lifetime (default: 30 minutes)
JWT_ISSUER="https://your-app.com"        # Token issuer for validation
JWT_AUDIENCE="your-app-id"               # Token audience for validation

# Development Only - Disable Auth (NEVER in production)
AUTH_DISABLED=false                      # Default: false
```

### 2. Generate JWT Secret Key

Generate a secure random key:

```bash
# Using Python
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Using OpenSSL
openssl rand -base64 32
```

### 3. Make Authenticated Requests

Include JWT token in `Authorization` header:

```bash
curl -X POST http://localhost:8000/api/items/ \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Item", "quantity": 10, "price": 99.99}'
```

## Authentication Architecture

### JWT Token Structure

Tokens contain the following claims:

```json
{
  "sub": "user_id",              // Subject (user identifier) - REQUIRED
  "exp": 1234567890,             // Expiration timestamp - REQUIRED
  "iat": 1234567890,             // Issued at timestamp
  "iss": "https://your-app.com", // Issuer (optional but recommended)
  "aud": "your-app-id",          // Audience (optional but recommended)
  "email": "user@example.com",   // User email (custom claim)
  "username": "johndoe",         // Username (custom claim)
  "roles": ["user", "admin"],    // User roles for RBAC (custom claim)
  "tenant_id": "tenant_123"      // Tenant ID for multi-tenancy (custom claim)
}
```

### User Model

The authenticated user is represented by the `User` model:

```python
from api.security import User

user = User(
    id="user_123",                  # Unique user identifier (from 'sub' claim)
    email="user@example.com",       # User email (optional)
    username="johndoe",             # Username (optional)
    roles=["user", "admin"],        # User roles (optional)
    is_active=True,                 # Account status (default: True)
    tenant_id="tenant_abc",         # Tenant ID for multi-tenancy (optional)
    metadata={}                     # Additional user data (optional)
)
```

### Protected vs Public Endpoints

**Protected Endpoints** (require authentication):
- `POST /api/items/` - Create item
- `PUT /api/items/{id}` - Update item
- `DELETE /api/items/{id}` - Delete item
- `POST /api/items/{id}/restore` - Restore deleted item

**Public Endpoints** (no authentication required):
- `GET /api/items/` - List items
- `GET /api/items/{id}` - Get item
- `GET /api/items/search/` - Search items
- `GET /api/items/stats/count` - Count items

## Integrating with External Identity Providers

### Auth0 Integration

1. **Configure Auth0 Application**:
   - Create an API in Auth0 dashboard
   - Note the Audience (API identifier)
   - Note the Domain (issuer)

2. **Update Environment Variables**:
   ```bash
   JWT_ALGORITHM="RS256"                              # Auth0 uses RS256
   JWT_ISSUER="https://your-tenant.auth0.com/"       # Your Auth0 domain
   JWT_AUDIENCE="https://api.your-app.com"           # Your API identifier
   JWT_SECRET_KEY="..."                              # Public key or JWKS URL
   ```

3. **Customize Token Validation** (if needed):
   
   Edit `api/security.py` to fetch JWKS from Auth0:
   ```python
   from jose import jwk, jwt
   import requests
   
   def get_auth0_public_key(token: str, jwks_url: str):
       """Fetch public key from Auth0 JWKS endpoint."""
       # Implementation to fetch and cache JWKS
       pass
   ```

### AWS Cognito Integration

1. **Configure Cognito User Pool**:
   - Create User Pool in AWS Cognito
   - Note the User Pool ID and Region
   - Note the App Client ID

2. **Update Environment Variables**:
   ```bash
   JWT_ALGORITHM="RS256"
   JWT_ISSUER="https://cognito-idp.{region}.amazonaws.com/{user_pool_id}"
   JWT_AUDIENCE="{app_client_id}"
   ```

3. **Customize Claims Mapping**:
   
   Edit `api/security.py` `token_data_to_user()` function:
   ```python
   def token_data_to_user(token_data: TokenData) -> User:
       """Map Cognito claims to User model."""
       return User(
           id=token_data.sub,
           email=token_data.email,
           username=token_data.get("cognito:username"),  # Cognito-specific claim
           roles=token_data.get("cognito:groups", []),   # Cognito groups
           # ...
       )
   ```

### Azure AD Integration

1. **Register Application in Azure AD**:
   - Register app in Azure portal
   - Note Application (client) ID
   - Note Directory (tenant) ID

2. **Update Environment Variables**:
   ```bash
   JWT_ALGORITHM="RS256"
   JWT_ISSUER="https://login.microsoftonline.com/{tenant_id}/v2.0"
   JWT_AUDIENCE="{client_id}"
   ```

3. **Customize for Azure AD Claims**:
   
   Azure AD uses different claim names:
   ```python
   def token_data_to_user(token_data: TokenData) -> User:
       """Map Azure AD claims to User model."""
       return User(
           id=token_data.sub or token_data.get("oid"),   # Azure AD object ID
           email=token_data.get("preferred_username"),    # Email in preferred_username
           username=token_data.get("name"),
           roles=token_data.get("roles", []),             # App roles
           # ...
       )
   ```

## Role-Based Access Control (RBAC)

### Using Roles in Routes

Create custom dependencies for role-based access:

```python
from fastapi import Depends, HTTPException, status
from api.dependencies import require_auth
from api.security import User

def require_admin(user: User = Depends(require_auth)) -> User:
    """Require admin role."""
    if "admin" not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user

def require_any_role(*required_roles: str):
    """Require any of the specified roles."""
    def _require_any_role(user: User = Depends(require_auth)) -> User:
        if not any(role in user.roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of: {', '.join(required_roles)}"
            )
        return user
    return _require_any_role

# Usage in routes
@router.post("/admin/action")
async def admin_action(user: User = Depends(require_admin)):
    return {"message": "Admin action performed"}

@router.post("/moderator/action")
async def moderator_action(
    user: User = Depends(require_any_role("admin", "moderator"))
):
    return {"message": "Moderator action performed"}
```

### Checking Roles Manually

```python
from api.security import verify_user_has_role

@router.post("/items/{item_id}/approve")
async def approve_item(
    item_id: UUID,
    user: User = Depends(require_auth)
):
    # Check if user has admin or moderator role
    if not (verify_user_has_role(user, "admin") or 
            verify_user_has_role(user, "moderator")):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Perform approval
    pass
```

## Testing Authentication

### Creating Test Tokens

Use `create_access_token` helper for testing:

```python
from api.security import create_access_token
from config.settings import get_settings

# Create test token
settings = get_settings()
token = create_access_token(
    user_id="test_user_123",
    settings=settings,
    email="test@example.com",
    username="testuser",
    roles=["user", "admin"],
    tenant_id="test_tenant",
)

# Use in tests
headers = {"Authorization": f"Bearer {token}"}
response = client.post("/api/items/", json=payload, headers=headers)
```

### Running Auth Tests

```bash
# Run all tests
pytest tests/

# Run auth-specific tests
pytest tests/test_security.py -v
pytest tests/test_auth_integration.py -v

# Run with coverage
pytest tests/test_security.py --cov=api/security --cov-report=html
```

## Security Best Practices

### 1. Secret Key Management

**✅ DO:**
- Use a cryptographically secure random key (minimum 32 characters)
- Store secret in environment variables, never in code
- Use different secrets for dev/staging/production
- Rotate secrets periodically

**❌ DON'T:**
- Hard-code secrets in source code
- Use weak or predictable secrets
- Commit secrets to version control
- Share secrets between environments

### 2. Token Lifetime

**Recommended:**
- Access tokens: 15-30 minutes
- Refresh tokens (if implemented): 7-30 days
- Shorter lifetimes for high-security operations

### 3. HTTPS Requirements

**⚠️ CRITICAL:** Always use HTTPS in production. JWT tokens in HTTP headers can be intercepted.

```python
# Enforce HTTPS in production
if settings.is_production:
    app.add_middleware(
        HTTPSRedirectMiddleware
    )
```

### 4. Token Validation

The system validates:
- ✅ Signature (prevents tampering)
- ✅ Expiration (prevents replay attacks)
- ✅ Issuer (prevents token from wrong source)
- ✅ Audience (prevents token for wrong app)

### 5. Error Handling

Authentication errors return proper HTTP status codes:
- `401 Unauthorized` - No token, invalid token, expired token
- `403 Forbidden` - Valid token but insufficient permissions

Error responses don't leak sensitive information:
```json
{
  "detail": "Authentication required"
}
```

## Development Mode

### Disabling Authentication (Local Development Only)

**⚠️ WARNING:** Only use in local development, never in production!

```bash
AUTH_DISABLED=true
```

When authentication is disabled:
- All routes become accessible without tokens
- `get_current_user()` returns `None`
- `require_auth()` still works but returns a mock user

The system **will refuse to start** if `AUTH_DISABLED=true` in production environment.

## Troubleshooting

### Token Expired Error

**Problem:** `401 Unauthorized - Token has expired`

**Solution:**
- Generate a new token
- Increase `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` if tokens expire too quickly
- Implement token refresh mechanism

### Invalid Token Signature

**Problem:** `401 Unauthorized - Invalid token signature`

**Solution:**
- Verify `JWT_SECRET_KEY` matches between token creation and validation
- Check `JWT_ALGORITHM` is consistent
- Ensure token wasn't modified in transit

### Invalid Issuer/Audience

**Problem:** `401 Unauthorized - Invalid token issuer/audience`

**Solution:**
- Verify `JWT_ISSUER` and `JWT_AUDIENCE` match token claims
- Set issuer/audience to `None` if not using validation
- Check identity provider configuration

### Missing Required Claims

**Problem:** `401 Unauthorized - Invalid token`

**Solution:**
- Ensure token includes required `sub` and `exp` claims
- Check identity provider includes necessary claims
- Customize token validation in `security.py`

## API Reference

### Security Module (`api/security.py`)

**Classes:**
- `User` - Authenticated user model
- `TokenData` - JWT token payload
- Exception classes: `AuthenticationError`, `TokenExpiredError`, `TokenInvalidError`, `TokenValidationError`

**Functions:**
- `decode_jwt_token(token, settings)` - Decode and validate JWT
- `token_data_to_user(token_data)` - Convert token to User
- `create_access_token(user_id, settings, ...)` - Create JWT token
- `verify_user_has_role(user, role)` - Check single role
- `verify_user_has_any_role(user, roles)` - Check any role
- `verify_user_has_all_roles(user, roles)` - Check all roles

### Dependencies (`api/dependencies.py`)

**Functions:**
- `get_current_user()` - Extract user from request (optional)
- `require_auth()` - Require authentication (raises 401 if missing)

## Support

For questions or issues:
1. Check environment variables configuration
2. Review server logs for detailed error messages
3. Consult identity provider documentation
4. Review `api/security.py` and `api/dependencies.py` for customization

## References

- [JWT Standard (RFC 7519)](https://tools.ietf.org/html/rfc7519)
- [FastAPI Security Documentation](https://fastapi.tiangolo.com/tutorial/security/)
- [PyJWT Library](https://pyjwt.readthedocs.io/)
