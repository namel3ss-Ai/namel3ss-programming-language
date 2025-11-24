"""Smoke tests to ensure scaffold/runtime assets ship with the repo and are populated."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_ASSETS = [
    # Frontend widget runtime
    "namel3ss/codegen/frontend/templates/widget-core.js",
    "namel3ss/codegen/frontend/templates/widget-rendering.js",
    "namel3ss/codegen/frontend/templates/widget-realtime.js",
    # CRUD project template
    "namel3ss/project_templates/crud_service/files/README.md",
    "namel3ss/project_templates/crud_service/files/GETTING_STARTED.md",
    "namel3ss/project_templates/crud_service/files/AUTHENTICATION.md",
    "namel3ss/project_templates/crud_service/files/app.n3",
    "namel3ss/project_templates/crud_service/files/requirements.txt",
    "namel3ss/project_templates/crud_service/files/requirements-dev.txt",
    "namel3ss/project_templates/crud_service/files/pytest.ini",
    "namel3ss/project_templates/crud_service/files/Dockerfile",
    "namel3ss/project_templates/crud_service/files/Makefile",
    "namel3ss/project_templates/crud_service/files/docker-compose.yml",
    "namel3ss/project_templates/crud_service/files/migrations.sql",
    "namel3ss/project_templates/crud_service/files/.env.example",
    "namel3ss/project_templates/crud_service/files/.gitignore",
    "namel3ss/project_templates/crud_service/files/config/.env.example",
]


def test_required_assets_exist_and_are_nonempty():
    missing = []
    empty = []
    for rel in REQUIRED_ASSETS:
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        if path.stat().st_size < 50:
            empty.append(rel)
    assert not missing, f"Missing assets: {missing}"
    assert not empty, f"Assets appear empty: {empty}"


def test_env_template_declares_security_and_db_keys():
    env_template = ROOT / "namel3ss/project_templates/crud_service/files/.env.example"
    text = env_template.read_text(encoding="utf-8")
    for key in ("DATABASE_URL", "JWT_SECRET_KEY", "LOG_LEVEL", "CORS_ORIGINS"):
        assert key in text, f"{key} should be present in .env.example"


def test_requirements_are_pinned():
    reqs = (ROOT / "namel3ss/project_templates/crud_service/files/requirements.txt").read_text(
        encoding="utf-8"
    )
    assert "==" in reqs, "Production requirements should be pinned"


def test_manifest_mentions_frontend_templates_and_envs():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "frontend/templates" in manifest
    assert ".env" in manifest
