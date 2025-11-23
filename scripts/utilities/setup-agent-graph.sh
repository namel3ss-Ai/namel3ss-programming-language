#!/bin/bash

# N3 Agent Graph Editor - Setup Script

set -e

echo "🚀 Setting up N3 Agent Graph Editor..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.11+ first."
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites met${NC}"

# Install frontend dependencies
echo -e "${BLUE}Installing frontend dependencies...${NC}"
cd src/web/graph-editor
npm install
cd ../../..
echo -e "${GREEN}✓ Frontend dependencies installed${NC}"

# Install backend dependencies
echo -e "${BLUE}Installing backend dependencies...${NC}"
cd n3_server
pip install -r requirements.txt
cd ..
echo -e "${GREEN}✓ Backend dependencies installed${NC}"

# Install Yjs server dependencies
echo -e "${BLUE}Installing Yjs server dependencies...${NC}"
cd yjs-server
npm install
cd ..
echo -e "${GREEN}✓ Yjs server dependencies installed${NC}"

# Start Docker services
echo -e "${BLUE}Starting Docker services...${NC}"
docker-compose up -d postgres jaeger
echo -e "${GREEN}✓ Docker services started${NC}"

# Wait for PostgreSQL
echo -e "${BLUE}Waiting for PostgreSQL...${NC}"
sleep 5

# Run database migrations
echo -e "${BLUE}Running database migrations...${NC}"
alembic upgrade head
echo -e "${GREEN}✓ Database migrations complete${NC}"

# Create initial demo project
echo -e "${BLUE}Creating demo project...${NC}"
python3 - <<EOF
import asyncio
from sqlalchemy import select
from n3_server.db.session import AsyncSessionLocal
from n3_server.db.models import Project

async def create_demo():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Project).where(Project.id == "demo"))
        existing = result.scalar_one_or_none()
        
        if not existing:
            demo = Project(
                id="demo",
                name="Demo Project",
                graph_data={
                    "chains": [],
                    "agents": [],
                    "activeRootId": "start-1",
                    "nodes": [
                        {
                            "id": "start-1",
                            "type": "start",
                            "label": "START",
                            "data": {},
                            "position": {"x": 100, "y": 200}
                        },
                        {
                            "id": "prompt-1",
                            "type": "prompt",
                            "label": "Welcome Prompt",
                            "data": {"text": "Hello! How can I help you today?"},
                            "position": {"x": 300, "y": 200}
                        },
                        {
                            "id": "end-1",
                            "type": "end",
                            "label": "END",
                            "data": {},
                            "position": {"x": 600, "y": 200}
                        }
                    ],
                    "edges": [
                        {"id": "e1", "source": "start-1", "target": "prompt-1"},
                        {"id": "e2", "source": "prompt-1", "target": "end-1"}
                    ]
                },
                metadata={}
            )
            session.add(demo)
            await session.commit()
            print("Demo project created")
        else:
            print("Demo project already exists")

asyncio.run(create_demo())
EOF
echo -e "${GREEN}✓ Demo project created${NC}"

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "To start the development servers:"
echo ""
echo "  Terminal 1 (Backend):"
echo "    cd n3_server"
echo "    uvicorn api.main:app --reload --port 8000"
echo ""
echo "  Terminal 2 (Yjs Server):"
echo "    cd yjs-server"
echo "    npm start"
echo ""
echo "  Terminal 3 (Frontend):"
echo "    cd src/web/graph-editor"
echo "    npm run dev"
echo ""
echo "Access points:"
echo "  Frontend:  http://localhost:3000"
echo "  API:       http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo "  Jaeger UI: http://localhost:16686"
echo ""
