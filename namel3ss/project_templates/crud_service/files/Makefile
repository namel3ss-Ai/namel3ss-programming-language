# {{ project_name }} - Production CRUD Service
# Makefile for common development and deployment tasks

.PHONY: help install install-dev setup db-setup db-migrate db-reset test test-unit test-integration test-coverage lint format run dev docker-build docker-run clean

# Default target
help:
	@echo "{{ project_name }} - Available Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make setup            Full setup (venv, install, db)"
	@echo ""
	@echo "Database:"
	@echo "  make db-setup         Create database and run migrations"
	@echo "  make db-migrate       Run database migrations"
	@echo "  make db-reset         Drop and recreate database"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-coverage    Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run linters"
	@echo "  make format           Format code"
	@echo ""
	@echo "Development:"
	@echo "  make run              Run production server"
	@echo "  make dev              Run development server with reload"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-run       Run Docker container"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove generated files"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

setup: install-dev db-setup
	@echo "Setup complete! Run 'make dev' to start development server."

# Database
db-setup:
	@echo "Creating database..."
	createdb {{ project_name | lower | replace('-', '_') }}_db || true
	@echo "Running migrations..."
	psql -U postgres -d {{ project_name | lower | replace('-', '_') }}_db -f migrations.sql
	@echo "Database setup complete!"

db-migrate:
	psql -U postgres -d {{ project_name | lower | replace('-', '_') }}_db -f migrations.sql

db-reset:
	@echo "Resetting database..."
	dropdb {{ project_name | lower | replace('-', '_') }}_db || true
	createdb {{ project_name | lower | replace('-', '_') }}_db
	psql -U postgres -d {{ project_name | lower | replace('-', '_') }}_db -f migrations.sql
	@echo "Database reset complete!"

# Testing
test:
	pytest tests/ -v

test-unit:
	pytest tests/ -v -m unit

test-integration:
	pytest tests/ -v -m integration

test-coverage:
	pytest tests/ --cov=. --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

# Code quality
lint:
	@echo "Running linters..."
	@command -v ruff >/dev/null 2>&1 && ruff check . || echo "ruff not installed (pip install ruff)"
	@command -v mypy >/dev/null 2>&1 && mypy . || echo "mypy not installed (pip install mypy)"

format:
	@echo "Formatting code..."
	@command -v ruff >/dev/null 2>&1 && ruff format . || echo "ruff not installed (pip install ruff)"
	@command -v isort >/dev/null 2>&1 && isort . || echo "isort not installed (pip install isort)"

# Development
run:
	uvicorn main:app --host 0.0.0.0 --port 8000

dev:
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker-build:
	docker build -t {{ project_name | lower | replace('_', '-') }}:latest .

docker-run:
	docker run -p 8000:8000 --env-file .env {{ project_name | lower | replace('_', '-') }}:latest

# Cleanup
clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "Cleanup complete!"
