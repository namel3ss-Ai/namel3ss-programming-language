# Authentication and Authorization Implementation

Complete OAuth2/JWT-based authentication system with role-based access control (RBAC) for N3 platform.

## Overview

Secure authentication and authorization system with:
- **JWT tokens** for stateless authentication
- **Role-based permissions** (Owner, Editor, Viewer)
- **Project ownership** and membership
- **Password hashing** with bcrypt
- **Token refresh** mechanism

## Architecture

```
User Registration → Email/Password → Hashed Password Stored
                                                ↓
User Login → Verify Credentials → JWT Access Token + Refresh Token
                                          ↓
Protected Endpoint → Validate Token → Check Permissions → Allow/Deny
```

## Database Schema

### User Model
```python
class User:
    id: str                    # Unique user ID
    email: str                 # Unique email (login)
    username: str              # Unique username (login)
    hashed_password: str       # Bcrypt hashed password
    full_name: str | None      # Optional full name
    is_active: bool            # Account status
    is_superuser: bool         # Admin flag
    created_at: datetime
    updated_at: datetime
    last_login: datetime | None
```

### ProjectMember Model
```python
class ProjectMember:
    id: str
    project_id: str            # ForeignKey to Project
    user_id: str               # ForeignKey to User
    role: Role                 # OWNER | EDITOR | VIEWER
    created_at: datetime
    updated_at: datetime
```

### Role Hierarchy
```
OWNER (3)   → Full control (delete, manage members, edit, view)
EDITOR (2)  → Edit and view
VIEWER (1)  → View only
```

## API Endpoints

### Authentication

#### POST /api/auth/register
Register a new user.

**Request**:
```json
{
  "email": "user@example.com",
  "username": "johndoe",
  "password": "strongpassword123",
  "full_name": "John Doe"
}
```

**Response (201)**:
```json
{
  "id": "abc123",
  "email": "user@example.com",
  "username": "johndoe",
  "full_name": "John Doe",
  "is_active": true,
  "is_superuser": false,
  "created_at": "2024-01-01T00:00:00Z",
  "last_login": null
}
```

**Validation**:
- Email must be unique and valid
- Username must be 3-50 characters, unique
- Password must be at least 8 characters

#### POST /api/auth/login
Login with username/email and password.

**Request**:
```json
{
  "username": "johndoe",  // or email
  "password": "strongpassword123"
}
```

**Response (200)**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Errors**:
- 401: Incorrect username or password
- 403: Inactive user

#### POST /api/auth/refresh
Refresh access token using refresh token.

**Request**:
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response (200)**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### GET /api/auth/me
Get current user information.

**Headers**:
```
Authorization: Bearer <access_token>
```

**Response (200)**:
```json
{
  "id": "abc123",
  "email": "user@example.com",
  "username": "johndoe",
  "full_name": "John Doe",
  "is_active": true,
  "is_superuser": false,
  "created_at": "2024-01-01T00:00:00Z",
  "last_login": "2024-01-02T10:30:00Z"
}
```

#### PATCH /api/auth/me
Update current user profile.

**Headers**:
```
Authorization: Bearer <access_token>
```

**Request**:
```json
{
  "full_name": "John Updated Doe",
  "email": "newemail@example.com"
}
```

**Response (200)**:
```json
{
  "id": "abc123",
  "email": "newemail@example.com",
  "username": "johndoe",
  "full_name": "John Updated Doe",
  ...
}
```

#### POST /api/auth/change-password
Change current user's password.

**Headers**:
```
Authorization: Bearer <access_token>
```

**Request**:
```json
{
  "current_password": "oldpassword123",
  "new_password": "newstrongpassword456"
}
```

**Response (204)**:
No content on success.

### Project Membership

#### GET /api/projects/{project_id}/members
List all members of a project.

**Requires**: Owner access

**Response (200)**:
```json
[
  {
    "id": "member_123",
    "user_id": "user_456",
    "user_email": "member@example.com",
    "user_username": "memberuser",
    "role": "editor",
    "created_at": "2024-01-01T00:00:00Z"
  }
]
```

#### POST /api/projects/{project_id}/members
Add a member to a project.

**Requires**: Owner access

**Request**:
```json
{
  "email": "newmember@example.com",
  "role": "editor"  // owner | editor | viewer
}
```

**Response (201)**:
```json
{
  "id": "member_789",
  "user_id": "user_012",
  "user_email": "newmember@example.com",
  "user_username": "newuser",
  "role": "editor",
  "created_at": "2024-01-03T00:00:00Z"
}
```

**Errors**:
- 404: User not found with that email
- 400: User is already a member

#### PATCH /api/projects/{project_id}/members/{member_id}
Update a member's role.

**Requires**: Owner access

**Request**:
```json
{
  "role": "viewer"
}
```

**Response (200)**:
```json
{
  "id": "member_789",
  "user_id": "user_012",
  "user_email": "member@example.com",
  "user_username": "memberuser",
  "role": "viewer",
  "created_at": "2024-01-03T00:00:00Z"
}
```

#### DELETE /api/projects/{project_id}/members/{member_id}
Remove a member from a project.

**Requires**: Owner access

**Response (204)**:
No content on success.

## Usage Examples

### Registration and Login

```python
import requests

# Register new user
response = requests.post(
    "http://localhost:8000/api/auth/register",
    json={
        "email": "alice@example.com",
        "username": "alice",
        "password": "securepass123",
        "full_name": "Alice Smith",
    }
)
user = response.json()

# Login
response = requests.post(
    "http://localhost:8000/api/auth/login",
    json={
        "username": "alice",
        "password": "securepass123",
    }
)
tokens = response.json()
access_token = tokens["access_token"]

# Use access token
headers = {"Authorization": f"Bearer {access_token}"}
response = requests.get(
    "http://localhost:8000/api/auth/me",
    headers=headers,
)
print(response.json())
```

### Project Membership Management

```python
# Add member to project
response = requests.post(
    f"http://localhost:8000/api/projects/{project_id}/members",
    headers=headers,
    json={
        "email": "bob@example.com",
        "role": "editor",
    }
)

# List members
response = requests.get(
    f"http://localhost:8000/api/projects/{project_id}/members",
    headers=headers,
)
members = response.json()

# Update member role
response = requests.patch(
    f"http://localhost:8000/api/projects/{project_id}/members/{member_id}",
    headers=headers,
    json={"role": "viewer"},
)

# Remove member
response = requests.delete(
    f"http://localhost:8000/api/projects/{project_id}/members/{member_id}",
    headers=headers,
)
```

### Token Refresh

```python
# Refresh access token before it expires
response = requests.post(
    "http://localhost:8000/api/auth/refresh",
    json={"refresh_token": refresh_token},
)
new_tokens = response.json()
access_token = new_tokens["access_token"]
```

## Protecting Endpoints

### Require Authentication

```python
from fastapi import APIRouter, Depends
from n3_server.auth import get_current_active_user, User

router = APIRouter()

@router.get("/protected")
async def protected_endpoint(
    current_user: User = Depends(get_current_active_user),
):
    return {"message": f"Hello {current_user.username}"}
```

### Require Project Access

```python
from n3_server.auth import require_editor_access, User

@router.patch("/projects/{project_id}/graphs")
async def update_graph(
    project_id: str,
    current_user: User = Depends(require_editor_access()),
):
    # User has editor or owner access
    return {"message": "Graph updated"}
```

### Access Levels

```python
from n3_server.auth import (
    require_viewer_access,   # Minimum: viewer
    require_editor_access,   # Minimum: editor  
    require_owner_access,    # Minimum: owner
)

# View project
@router.get("/projects/{project_id}")
async def get_project(
    project_id: str,
    current_user: User = Depends(require_viewer_access()),
):
    pass

# Edit project
@router.patch("/projects/{project_id}")
async def update_project(
    project_id: str,
    current_user: User = Depends(require_editor_access()),
):
    pass

# Manage project
@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: str,
    current_user: User = Depends(require_owner_access()),
):
    pass
```

### Optional Authentication

```python
from n3_server.auth import get_optional_user

@router.get("/public-or-private")
async def mixed_endpoint(
    user: User | None = Depends(get_optional_user),
):
    if user:
        return {"message": f"Hello {user.username}"}
    else:
        return {"message": "Hello anonymous"}
```

## Security Features

### Password Hashing
- **Algorithm**: bcrypt with salt
- **Library**: passlib
- **Security**: Industry-standard secure password storage

### JWT Tokens
- **Algorithm**: HS256 (HMAC with SHA-256)
- **Library**: python-jose
- **Access Token Lifetime**: 30 minutes
- **Refresh Token Lifetime**: 7 days

### Token Types
- **Access Token**: Short-lived, for API access
- **Refresh Token**: Long-lived, for getting new access tokens
- **Type Validation**: Prevents using refresh tokens as access tokens

### Protection Against
- **SQL Injection**: Parameterized queries via SQLAlchemy
- **Password Cracking**: Bcrypt hashing with automatic salting
- **Token Replay**: Token expiration and type validation
- **Brute Force**: Account lockout (TODO: implement rate limiting)

## Configuration

### Environment Variables

```bash
# Secret key for JWT signing (MUST be changed in production)
SECRET_KEY=your-super-secret-key-min-32-chars

# Database connection
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/n3_db

# Token lifetimes (optional, defaults shown)
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
```

### Generate Secure Secret Key

```python
import secrets

# Generate a secure random secret key
secret_key = secrets.token_urlsafe(32)
print(f"SECRET_KEY={secret_key}")
```

Or use command line:
```bash
openssl rand -hex 32
```

## Testing

Run auth tests:

```bash
# All auth tests
pytest test_auth.py -v

# Specific test class
pytest test_auth.py::TestUserRegistration -v

# Specific test
pytest test_auth.py::TestUserLogin::test_login_with_username -v
```

Test coverage:
- User registration (duplicate email/username, validation)
- User login (username/email, wrong password)
- Token management (refresh, expiration)
- User profile (get, update, change password)
- Project permissions (add/list/update/remove members)

## Migration from Unauthenticated System

### Backward Compatibility

The Project model includes `owner_id` as **nullable** to support migration:

```python
class Project(Base):
    owner_id = Column(String(50), ForeignKey("users.id"), nullable=True)
```

### Migration Steps

1. **Add User Table**:
```bash
# Run database migration
alembic revision --autogenerate -m "Add authentication tables"
alembic upgrade head
```

2. **Create Default Admin User**:
```python
from n3_server.auth import get_password_hash
from n3_server.auth.models import User

admin = User(
    id="admin_001",
    email="admin@n3.local",
    username="admin",
    hashed_password=get_password_hash("admin_password"),
    is_active=True,
    is_superuser=True,
)
db.add(admin)
await db.commit()
```

3. **Assign Owners to Existing Projects**:
```sql
-- Assign all projects to admin
UPDATE projects SET owner_id = 'admin_001' WHERE owner_id IS NULL;
```

4. **Make owner_id Required** (optional):
```python
# After migration, make owner_id required
class Project(Base):
    owner_id = Column(String(50), ForeignKey("users.id"), nullable=False)
```

## Best Practices

### Password Requirements
- Minimum 8 characters
- Mix of uppercase, lowercase, numbers, symbols (enforced by frontend)
- Not same as username or email

### Token Management
- Store refresh tokens securely (httpOnly cookies recommended)
- Implement token revocation for logout (TODO)
- Use short-lived access tokens (30 min)

### Permission Checks
- Always check permissions server-side
- Never trust client-side permission checks
- Log permission denials for security monitoring

### User Management
- Implement email verification (TODO)
- Add password reset functionality (TODO)
- Implement account lockout after failed attempts (TODO)
- Add two-factor authentication (TODO)

## Troubleshooting

### "Could not validate credentials"
- Token expired → refresh token
- Invalid token → login again
- User inactive → contact admin

### "Insufficient permissions"
- Check user role: GET /api/auth/me
- Verify project membership: GET /api/projects/{id}/members
- Contact project owner for access

### "Email already registered"
- Use different email
- Or recover existing account

### "User not found with that email"
- User must register first before being added to project
- Check email spelling

## Future Enhancements

1. **Email Verification**: Send verification email on registration
2. **Password Reset**: Email-based password reset flow
3. **OAuth2 Providers**: Google, GitHub, Microsoft login
4. **Two-Factor Authentication**: TOTP-based 2FA
5. **API Key Authentication**: For programmatic access
6. **Rate Limiting**: Prevent brute force attacks
7. **Audit Logging**: Track all auth events
8. **Account Lockout**: Temporary lockout after failed attempts
9. **Session Management**: List and revoke active sessions
10. **RBAC Extensions**: Custom roles beyond Owner/Editor/Viewer

## API Client Examples

### Python

```python
from dataclasses import dataclass
import requests

@dataclass
class N3Client:
    base_url: str
    access_token: str | None = None
    refresh_token: str | None = None
    
    def login(self, username: str, password: str):
        response = requests.post(
            f"{self.base_url}/api/auth/login",
            json={"username": username, "password": password},
        )
        tokens = response.json()
        self.access_token = tokens["access_token"]
        self.refresh_token = tokens["refresh_token"]
    
    def _headers(self):
        return {"Authorization": f"Bearer {self.access_token}"}
    
    def get_projects(self):
        response = requests.get(
            f"{self.base_url}/api/projects",
            headers=self._headers(),
        )
        return response.json()

# Usage
client = N3Client("http://localhost:8000")
client.login("alice", "password123")
projects = client.get_projects()
```

### JavaScript/TypeScript

```typescript
class N3Client {
  private accessToken?: string;
  private refreshToken?: string;
  
  constructor(private baseUrl: string) {}
  
  async login(username: string, password: string) {
    const response = await fetch(`${this.baseUrl}/api/auth/login`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username, password}),
    });
    const tokens = await response.json();
    this.accessToken = tokens.access_token;
    this.refreshToken = tokens.refresh_token;
  }
  
  private headers() {
    return {
      'Authorization': `Bearer ${this.accessToken}`,
      'Content-Type': 'application/json',
    };
  }
  
  async getProjects() {
    const response = await fetch(`${this.baseUrl}/api/projects`, {
      headers: this.headers(),
    });
    return response.json();
  }
}

// Usage
const client = new N3Client('http://localhost:8000');
await client.login('alice', 'password123');
const projects = await client.getProjects();
```

## License

Same as main N3 project license.
