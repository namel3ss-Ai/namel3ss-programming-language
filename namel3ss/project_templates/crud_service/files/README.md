# {{ project_name }}

Production-grade CRUD microservice template built with **Namel3ss (N3)**, FastAPI, and PostgreSQL.

## Overview

This is a fully-functional RESTful API service demonstrating N3's capabilities as a general-purpose platform for building production-ready applications. It provides complete CRUD operations with pagination, filtering, soft deletes, and multi-tenancy support.

**Key Features:**
- ✅ Full CRUD operations (Create, Read, Update, Delete, Restore)
- ✅ PostgreSQL with connection pooling (asyncpg)
- ✅ Soft delete support
- ✅ Pagination and filtering
- ✅ Search functionality
- ✅ Multi-tenancy ready
- ✅ OpenAPI/Swagger documentation
- ✅ Structured logging with request IDs
- ✅ Health checks
- ✅ CORS support
- ✅ Error handling with consistent responses
- ✅ Repository pattern for clean architecture
- ✅ N3 DSL integration for declarative configuration

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- pip or poetry

### Installation

1. **Clone or generate this template:**
   ```bash
   n3 init --template crud-service --name my-service
   cd my-service
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up database:**
   ```bash
   # Create PostgreSQL database
   createdb {{ project_name | lower | replace('-', '_') }}_db
   
   # Run migrations
   psql -U postgres -d {{ project_name | lower | replace('-', '_') }}_db -f migrations.sql
   ```

5. **Configure environment:**
   ```bash
   cp config/.env.example .env
   # Edit .env with your settings (especially DATABASE_URL)
   ```

6. **Run the service:**
   ```bash
   python main.py
   ```

The API will be available at http://localhost:8000

- **API Documentation:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

## Architecture

```
{{ project_name }}/
├── app.ai                 # N3 DSL configuration (declarative)
├── main.py                # FastAPI application entry point
├── n3_integration.py      # N3 runtime integration
├── requirements.txt       # Python dependencies
├── migrations.sql         # Database schema
├── config/
│   ├── settings.py        # Application settings (Pydantic)
│   └── .env.example       # Environment template
├── models/
│   ├── domain.py          # Core business entities
│   └── schemas.py         # API request/response DTOs
├── repository/
│   ├── interface.py       # Repository contract (ABC)
│   └── postgres.py        # PostgreSQL implementation
└── api/
    ├── routes.py          # HTTP endpoints
    ├── dependencies.py    # Dependency injection
    └── errors.py          # Error handling
```

### Design Patterns

- **Repository Pattern:** Data access abstraction for testability
- **Dependency Injection:** Loose coupling via FastAPI's DI system
- **Domain-Driven Design:** Clear separation between domain and infrastructure
- **Configuration as Code:** N3 DSL for declarative configuration

## API Endpoints

### CRUD Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/{{ endpoint_prefix }}/` | Create new item |
| `GET` | `/{{ endpoint_prefix }}/{id}` | Get item by ID |
| `GET` | `/{{ endpoint_prefix }}/` | List items (paginated) |
| `GET` | `/{{ endpoint_prefix }}/search/` | Search by name |
| `PUT` | `/{{ endpoint_prefix }}/{id}` | Update item |
| `DELETE` | `/{{ endpoint_prefix }}/{id}` | Delete item (soft) |
| `POST` | `/{{ endpoint_prefix }}/{id}/restore` | Restore deleted item |
| `GET` | `/{{ endpoint_prefix }}/stats/count` | Count items |

### Examples

**Create Item:**
```bash
curl -X POST http://localhost:8000/{{ endpoint_prefix }}/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Premium Widget",
    "description": "High-quality widget",
    "quantity": 100,
    "price": 49.99,
    "tags": ["electronics", "premium"],
    "metadata": {"color": "silver"}
  }'
```

**List Items with Filtering:**
```bash
curl "http://localhost:8000/{{ endpoint_prefix }}/?page=1&page_size=20&is_active=true&tags=electronics"
```

**Search Items:**
```bash
curl "http://localhost:8000/{{ endpoint_prefix }}/search/?q=widget"
```

**Update Item:**
```bash
curl -X PUT http://localhost:8000/{{ endpoint_prefix }}/{id} \
  -H "Content-Type: application/json" \
  -d '{"price": 59.99, "is_active": false}'
```

## Configuration

### Environment Variables

Key variables (see `config/.env.example` for full list):

```bash
# Required
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname

# Optional
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=true
CORS_ORIGINS=["http://localhost:3000"]
ENABLE_MULTI_TENANCY=false
```

### N3 Configuration

The `app.ai` file provides declarative configuration:

```n3
dataset {{ dataset_name }} {
    name: "{{ entity_name }} Management"
    schema {
        id: uuid(primary_key: true)
        name: string(required: true)
        // ... more fields
    }
}

api {{ api_name }} {
    dataset: {{ dataset_name }}
    operations { create, read, update, delete, list, search }
    pagination { default_page_size: 20, max_page_size: 100 }
}
```

## Extension Points

### 1. Add Authentication

Edit `api/dependencies.py`:
```python
async def get_current_user(request: Request) -> dict:
    # Implement JWT validation
    token = request.headers.get("Authorization")
    payload = decode_jwt(token)
    return {"user_id": payload["sub"]}
```

Add to routes:
```python
@router.post("/", dependencies=[Depends(require_auth)])
async def create_item(...):
    ...
```

### 2. Enable Multi-Tenancy

Set in `.env`:
```bash
ENABLE_MULTI_TENANCY=true
TENANT_HEADER=X-Tenant-ID
```

Pass header in requests:
```bash
curl -H "X-Tenant-ID: tenant123" http://localhost:8000/{{ endpoint_prefix }}/
```

### 3. Add Custom Validation

Edit `models/domain.py`:
```python
def update_fields(self, **kwargs):
    # Custom business logic
    if kwargs.get('quantity', 0) > 1000:
        raise ValueError("Quantity cannot exceed 1000")
    # ... existing code
```

### 4. Add New Endpoints

1. Add method to `repository/interface.py`
2. Implement in `repository/postgres.py`
3. Add route in `api/routes.py`

## Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### With Gunicorn

```bash
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

### Environment Checklist

- [ ] Set `ENVIRONMENT=production`
- [ ] Set `DEBUG=false`
- [ ] Use strong `DATABASE_URL` with SSL
- [ ] Configure `CORS_ORIGINS` to allowed domains
- [ ] Set appropriate `LOG_LEVEL` (INFO or WARNING)
- [ ] Increase `workers` based on CPU cores
- [ ] Enable connection pooling (already configured)
- [ ] Set up monitoring and health checks
- [ ] Configure database backups

## Testing

```bash
# Install dev dependencies
pip install pytest pytest-asyncio httpx pytest-cov

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## Database Schema

The service uses PostgreSQL with the following schema:

- **UUID primary keys** (uuid-ossp extension)
- **Soft delete support** (deleted_at timestamp)
- **Automatic timestamps** (created_at, updated_at with trigger)
- **Multi-tenancy** (tenant_id field)
- **JSON metadata** (JSONB with GIN index)
- **Array tags** (TEXT[] with GIN index)
- **Constraints** (quantity >= 0, price >= 0)

See `migrations.sql` for full schema.

## Troubleshooting

**Database connection errors:**
- Verify `DATABASE_URL` format: `postgresql://user:pass@host:port/dbname`
- Ensure PostgreSQL is running
- Check network connectivity and firewall rules

**Import errors:**
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

**Migration errors:**
- Check PostgreSQL version (14+ recommended)
- Ensure uuid-ossp extension is available
- Run as superuser if needed for extension creation

## License

Generated by Namel3ss. Customize as needed for your project.

## Support

- **Namel3ss Documentation:** https://namel3ss.dev
- **FastAPI Documentation:** https://fastapi.tiangolo.com
- **PostgreSQL Documentation:** https://postgresql.org/docs

---

**Built with Namel3ss** - Demonstrating N3 as a production-grade platform for modern application development.
