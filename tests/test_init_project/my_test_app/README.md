# my_test_app

A Namel3ss application.

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL (optional, for database features)

### Installation

1. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Install Node dependencies:
```bash
cd frontend
npm install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Development

1. Build the application:
```bash
namel3ss build app.ai
```

2. Run the development server:
```bash
namel3ss run app.ai
```

The backend will be available at http://localhost:8000
The frontend will be available at http://localhost:5173

### Project Structure

```
my_test_app/
  app.ai                  # Main Namel3ss application definition
  backend/
    main.py              # FastAPI entrypoint
    requirements.txt     # Python dependencies
  frontend/
    src/
      App.tsx            # Main React component
      main.tsx           # React entrypoint
    package.json         # Node dependencies
    vite.config.ts       # Vite configuration
  .env.example           # Environment variable template
  README.md              # This file
```

### Updating Dependencies

After modifying `app.ai`, run:
```bash
namel3ss sync-deps
```

This will automatically update `requirements.txt` and `package.json` based on
features used in your application.

## Learn More

- [Namel3ss Documentation](https://github.com/namel3ss-Ai/namel3ss-programming-language)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
