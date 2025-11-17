# Changelog

All notable changes to this project will be documented in this file. The format roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to [Semantic Versioning](https://semver.org/).

## [0.4.0] - 2025-11-16

### Added
- Production-ready deployment artefacts are now emitted with every backend build, including a multi-stage Dockerfile, `.dockerignore`, reverse-proxy templates (nginx + Caddy), a CI starter workflow, and Terraform/Helm skeletons with environment placeholders.
- Observability primitives expanded to cover structured logging (pretty in dev, JSON in production), correlation IDs, `/healthz`, `/readyz`, and `/metrics` endpoints plus optional OpenTelemetry tracing hooks behind the `observability` extra.

### Changed
- Adopted Semantic Versioning moving forward; bumped the package to `0.4.0` and tightened optional extras so users can `pip install namel3ss[extras]` without pulling unwanted dependencies.
- Refresh the packaging metadata to remove local environment references, ensuring editable installs (`pip install -e .[dev]`) work cleanly on any machine.

### Fixed
- Hardened request logging and metric rendering to avoid crashes when instrumentation dependencies are missing, returning actionable warnings instead of stack traces.

## [0.3.0] - 2025-11-16

### Added
- Structured extras (`realtime`, `redis`, `ai-connectors`, `observability`, `dev`) to keep the core install minimal while making optional features explicit.
- Project-wide `CHANGELOG.md` and exposed `namel3ss.__version__` so `namel3ss --version` reports the installed build.
- Auto-generated backend deployment artefacts (Dockerfile, reverse-proxy templates, CI starter, Terraform + Helm skeletons) emitted alongside each backend build.
- Production-ready observability primitives: JSON logging, request/connector/dataset metrics with a `/metrics` endpoint, `/healthz` and `/readyz` checks, and optional OpenTelemetry tracing hooks.

### Changed
- Adopted Semantic Versioning starting with `0.3.0`; downstream users should pin compatible minor versions when depending on generated artefacts.
- Core runtime now depends on SQLAlchemy directly to avoid hidden imports and improve ergonomics for generated backends.

### Fixed
- Eliminated references to local virtual environments in packaging metadata to ensure `pip install namel3ss[extras...]` works in clean environments.

## [0.2.0] - 2024-07-01

- Previous beta release focused on language features, realtime scaffolding, and baseline FastAPI generation.
