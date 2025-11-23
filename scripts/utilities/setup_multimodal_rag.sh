#!/bin/bash
# Quick setup script for Multimodal RAG system

set -e

echo "🚀 Namel3ss Multimodal RAG Setup"
echo "================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install base requirements
echo "Installing base requirements..."
pip install -q --upgrade pip

# Install Namel3ss package
echo "Installing Namel3ss package..."
pip install -q -e .

# Install multimodal requirements
echo "Installing multimodal RAG dependencies..."
pip install -q -r requirements-multimodal.txt

# Install dev requirements for testing
echo "Installing development dependencies..."
pip install -q -r requirements-dev.txt

echo ""
echo "✅ Installation complete!"
echo ""

# Check if Qdrant is running
echo "Checking Qdrant availability..."
if curl -s http://localhost:6333 > /dev/null 2>&1; then
    echo "✓ Qdrant is running on localhost:6333"
else
    echo "⚠️  Qdrant is not running. Start it with:"
    echo "   docker run -p 6333:6333 qdrant/qdrant"
fi

echo ""
echo "🎯 Quick Start:"
echo ""
echo "1. Start Qdrant (if not running):"
echo "   docker run -p 6333:6333 qdrant/qdrant"
echo ""
echo "2. Run example N3 application:"
echo "   n3 run examples/multimodal_rag.n3"
echo ""
echo "3. Start FastAPI service:"
echo "   uvicorn api.main:app --reload"
echo ""
echo "4. Run tests:"
echo "   pytest -v"
echo ""
echo "5. Run evaluation:"
echo "   n3 eval rag your_dataset.json --use-llm-judge"
echo ""
echo "📚 Documentation:"
echo "   - User Guide: MULTIMODAL_RAG_GUIDE.md"
echo "   - API Reference: MULTIMODAL_RAG_API.md"
echo "   - Testing Guide: MULTIMODAL_RAG_TESTING_GUIDE.md"
echo "   - Implementation Summary: MULTIMODAL_RAG_IMPLEMENTATION_SUMMARY.md"
echo ""
echo "Happy coding! 🎉"
