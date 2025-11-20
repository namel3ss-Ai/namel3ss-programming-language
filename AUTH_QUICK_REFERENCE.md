# Authentication Quick Reference

## Quick Start

### 1. Register User

```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "alice@example.com",
    "username": "alice",
    "password": "securepass123",
    "full_name": "Alice Smith"
  }'
```

### 2. Login

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "alice",
    "password": "securepass123"
  }'
```

Response:
```json
{
  "access_token": "eyJhbGci...",
  "refresh_token": "eyJhbGci...",
  "token_type": "bearer"
}
```

### 3. Access Protected Endpoint

```bash
TOKEN="eyJhbGci..."
curl http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

### 4. Add Project Member

```bash
curl -X POST http://localhost:8000/api/projects/PROJECT_ID/members \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "bob@example.com",
    "role": "editor"
  }'
```

## Roles

| Role | Permissions |
|------|-------------|
| **Owner** | Full control: delete project, manage members, edit, view |
| **Editor** | Edit and view project |
| **Viewer** | View project only |

## API Endpoints

### Authentication

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/auth/register` | Register new user | No |
| POST | `/api/auth/login` | Login | No |
| POST | `/api/auth/refresh` | Refresh token | No (refresh token) |
| GET | `/api/auth/me` | Get current user | Yes |
| PATCH | `/api/auth/me` | Update profile | Yes |
| POST | `/api/auth/change-password` | Change password | Yes |

### Project Members

| Method | Endpoint | Description | Required Role |
|--------|----------|-------------|---------------|
| GET | `/api/projects/{id}/members` | List members | Owner |
| POST | `/api/projects/{id}/members` | Add member | Owner |
| PATCH | `/api/projects/{id}/members/{mid}` | Update role | Owner |
| DELETE | `/api/projects/{id}/members/{mid}` | Remove member | Owner |

## Password Requirements

- **Minimum length**: 8 characters
- **Recommended**: Mix of uppercase, lowercase, numbers, symbols

## Token Lifetimes

- **Access Token**: 30 minutes
- **Refresh Token**: 7 days

## Python Usage

### Simple Client

```python
import requests

class N3Client:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.access_token = None
    
    def register(self, email, username, password, full_name=None):
        response = requests.post(
            f"{self.base_url}/api/auth/register",
            json={
                "email": email,
                "username": username,
                "password": password,
                "full_name": full_name,
            }
        )
        return response.json()
    
    def login(self, username, password):
        response = requests.post(
            f"{self.base_url}/api/auth/login",
            json={"username": username, "password": password},
        )
        tokens = response.json()
        self.access_token = tokens["access_token"]
        return tokens
    
    def headers(self):
        return {"Authorization": f"Bearer {self.access_token}"}
    
    def get_me(self):
        response = requests.get(
            f"{self.base_url}/api/auth/me",
            headers=self.headers(),
        )
        return response.json()
    
    def add_member(self, project_id, email, role="viewer"):
        response = requests.post(
            f"{self.base_url}/api/projects/{project_id}/members",
            headers=self.headers(),
            json={"email": email, "role": role},
        )
        return response.json()

# Usage
client = N3Client()
client.register("alice@example.com", "alice", "pass123", "Alice")
client.login("alice", "pass123")
user = client.get_me()
print(f"Logged in as: {user['username']}")
```

## FastAPI Integration

### Protect Endpoint

```python
from fastapi import APIRouter, Depends
from n3_server.auth import get_current_active_user, User

router = APIRouter()

@router.get("/protected")
async def protected_route(
    current_user: User = Depends(get_current_active_user),
):
    return {"message": f"Hello {current_user.username}"}
```

### Require Project Access

```python
from n3_server.auth import require_editor_access

@router.patch("/projects/{project_id}")
async def update_project(
    project_id: str,
    current_user: User = Depends(require_editor_access()),
):
    # User has editor or owner access
    return {"status": "updated"}
```

### Access Levels

```python
from n3_server.auth import (
    require_viewer_access,   # View only
    require_editor_access,   # Edit + View
    require_owner_access,    # Full control
)
```

### Optional Auth

```python
from n3_server.auth import get_optional_user

@router.get("/public")
async def public_route(
    user: User | None = Depends(get_optional_user),
):
    if user:
        return {"message": f"Hello {user.username}"}
    return {"message": "Hello anonymous"}
```

## Configuration

### Environment Variables

```bash
# Required: Secret key for JWT (change in production!)
SECRET_KEY=your-secret-key-min-32-chars

# Optional: Token lifetimes
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/n3_db
```

### Generate Secret Key

```bash
# Method 1: Python
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Method 2: OpenSSL
openssl rand -hex 32
```

## Error Codes

| Code | Meaning | Solution |
|------|---------|----------|
| 401 | Unauthorized | Login again, token expired |
| 403 | Forbidden | Insufficient permissions |
| 400 | Bad Request | Check request data |
| 404 | Not Found | Resource doesn't exist |
| 422 | Validation Error | Fix request format |

## Common Errors

### "Could not validate credentials"
```bash
# Token expired - refresh it
curl -X POST http://localhost:8000/api/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "YOUR_REFRESH_TOKEN"}'
```

### "Insufficient permissions"
```bash
# Check your role
curl http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer $TOKEN"

# Check project members
curl http://localhost:8000/api/projects/PROJECT_ID/members \
  -H "Authorization: Bearer $TOKEN"
```

### "Email already registered"
```bash
# Login instead of registering
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "alice@example.com",
    "password": "yourpassword"
  }'
```

## Testing

```bash
# Run all auth tests
pytest test_auth.py -v

# Run specific test
pytest test_auth.py::TestUserLogin::test_login_with_username -v

# With coverage
pytest test_auth.py --cov=n3_server.auth --cov-report=html
```

## Database Migration

```bash
# Create migration
alembic revision --autogenerate -m "Add authentication tables"

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Security Checklist

- [ ] Change `SECRET_KEY` in production
- [ ] Use HTTPS in production
- [ ] Set strong password requirements
- [ ] Implement rate limiting (TODO)
- [ ] Add email verification (TODO)
- [ ] Enable password reset (TODO)
- [ ] Add audit logging (TODO)
- [ ] Implement 2FA (TODO)

## JavaScript/TypeScript Client

```typescript
class N3AuthClient {
  private accessToken?: string;
  
  constructor(private baseUrl: string = 'http://localhost:8000') {}
  
  async register(data: {
    email: string;
    username: string;
    password: string;
    full_name?: string;
  }) {
    const response = await fetch(`${this.baseUrl}/api/auth/register`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data),
    });
    return response.json();
  }
  
  async login(username: string, password: string) {
    const response = await fetch(`${this.baseUrl}/api/auth/login`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username, password}),
    });
    const tokens = await response.json();
    this.accessToken = tokens.access_token;
    return tokens;
  }
  
  private headers() {
    return {
      'Authorization': `Bearer ${this.accessToken}`,
      'Content-Type': 'application/json',
    };
  }
  
  async getMe() {
    const response = await fetch(`${this.baseUrl}/api/auth/me`, {
      headers: this.headers(),
    });
    return response.json();
  }
  
  async addMember(projectId: string, email: string, role: string = 'viewer') {
    const response = await fetch(
      `${this.baseUrl}/api/projects/${projectId}/members`,
      {
        method: 'POST',
        headers: this.headers(),
        body: JSON.stringify({email, role}),
      }
    );
    return response.json();
  }
}

// Usage
const client = new N3AuthClient();
await client.register({
  email: 'alice@example.com',
  username: 'alice',
  password: 'secure123',
});
await client.login('alice', 'secure123');
const user = await client.getMe();
console.log(`Logged in as: ${user.username}`);
```

## Token Storage

### Browser (Frontend)

```javascript
// Store tokens securely
localStorage.setItem('access_token', tokens.access_token);
localStorage.setItem('refresh_token', tokens.refresh_token);

// Retrieve token
const token = localStorage.getItem('access_token');

// Clear on logout
localStorage.removeItem('access_token');
localStorage.removeItem('refresh_token');
```

**Better**: Use httpOnly cookies for refresh tokens (requires backend changes).

### Python (Backend)

```python
# Store in memory/session
class AuthSession:
    def __init__(self):
        self.access_token = None
        self.refresh_token = None
        self.expires_at = None

session = AuthSession()
session.access_token = tokens['access_token']
```

## Workflow Examples

### New User Workflow

1. Register → 2. Login → 3. Access protected resources

### Project Collaboration Workflow

1. Owner creates project
2. Owner adds members with roles
3. Members login and access project based on role
4. Owner can update roles or remove members

### Token Refresh Workflow

1. Access token expires after 30 min
2. Use refresh token to get new access token
3. Continue using new access token
4. Refresh token expires after 7 days → login again

## Support

- Full documentation: `AUTH_IMPLEMENTATION.md`
- Test examples: `test_auth.py`
- API reference: See main documentation

## License

Same as N3 project license.
