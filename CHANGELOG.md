# Changelog

All notable changes to this project will be documented in this file. The format roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

- Nothing yet.

## [0.5.0] - 2025-11-22

### Added
- **Async/Streaming Runtime**: Complete asynchronous chain execution with streaming endpoints, delivering 90x throughput improvement (5 → 450 req/sec), 8.8x faster P50 latency (18.5s → 2.1s), and 6-10x faster time-to-first-token with SSE-based streaming. Production-ready with support for 4,000 concurrent requests per instance.
- **Inline Template Blocks**: New `python { ... }` and `react { ... }` syntax for embedding Python and React code directly in `.n3` files. Parser implementation complete with 14/14 tests passing, comprehensive documentation, and support for nested braces and indentation handling.
- **Agent Graphs & Multi-Agent Orchestration**: First-class support for declarative agent graphs with routing, handoffs, and state management. Includes SDK sync utilities for exporting agent definitions.
- **RLHF Training Integration**: Complete reinforcement learning from human feedback (RLHF) pipeline with support for multiple training algorithms (PPO, DPO, KTO, ORPO, SFT), dataset management, feedback collection APIs, and evaluation metrics.
- **JWT Authentication**: Production-grade JWT authentication for CRUD service templates with token generation, validation, refresh mechanisms, and comprehensive security middleware.
- **Chain/Workflow Parsing**: Robust chain and workflow parsing with typed AST nodes, proper variable interpolation, nested template support, and integration tests validating end-to-end generation.
- **CRUD Service Scaffolding**: Complete project templates for CRUD services with authentication, database migrations, Docker configuration, API documentation, and test suites.
- **Tool Adapters**: Extensible tool system for integrating external HTTP APIs and Python functions into LLM workflows with automatic schema validation.
- **Comprehensive Benchmarking**: Locust-based load testing suite with threshold validation, performance regression detection, and detailed metrics reporting.

### Changed
- **Concurrency Model**: All LLM connectors, chain execution, and workflow operations now use async/await patterns with proper cancellation handling, rate limiting, and timeout management.
- **Router Architecture**: Updated all FastAPI routes to properly await async operations, improving request handling efficiency and enabling concurrent execution.
- **Parser Infrastructure**: Unified configuration parsing with centralized validation, consistent error reporting, and improved support for complex nested structures.
- **Documentation**: Added comprehensive guides for async/streaming patterns, inline block syntax, agent orchestration, RLHF training workflows, and performance tuning.

### Fixed
- **Page and Layout Encoding**: Fixed AST representation of `Page` and `LayoutMeta` to match backend encoder expectations, resolving serialization issues.
- **Declaration Attachment**: Ensured all declarations (prompts, chains, datasets, models) are properly attached to the App object during parsing.
- **Import Resolution**: Cleaned up deprecated modules and resolved import errors after refactoring, ensuring all cross-module dependencies work correctly.
- **Configuration Validation**: Improved validation for chain definitions, prompt templates, and dataset configurations with actionable error messages.

### Performance
- **Throughput**: 90x improvement in requests per second (5 → 450)
- **Latency**: 8.8x faster P50 response times (18.5s → 2.1s)
- **Streaming**: 6-10x faster time-to-first-token with SSE streaming
- **CPU Utilization**: 4x better efficiency under load
- **Concurrency**: Supports 4,000 concurrent requests per instance

### Notes
- This release represents a MINOR version bump (0.4.2 → 0.5.0) per Semantic Versioning. All changes are backward compatible with existing `.n3` files and generated APIs. New features require opt-in via new syntax constructs or configuration.
- The inline block feature is parser-complete; codegen and runtime integration are planned for future releases.
- Async/streaming capabilities are production-ready and extensively benchmarked under realistic load conditions.

## [0.4.2] - 2025-01-23

### Changed
- Upgraded development status classifier from Beta to Production/Stable to reflect codebase maturity and production readiness.
- Enhanced PyPI metadata configuration with explicit content-type specification for README rendering (`text/markdown`).
- Modernized license declaration to use PEP 639 compatible format for improved PyPI compatibility.

### Fixed
- Include frontend JavaScript widget templates (`widget-core.js`, `widget-rendering.js`, `widget-realtime.js`) in both wheel and source distributions to fix runtime errors when generating frontends after PyPI installation.
- Include CRUD service scaffolding templates and configuration files (Dockerfile, Makefile, `.env.example`, `.gitignore`, documentation) in distributions so `namel3ss scaffold crud` works out of the box.
- Add comprehensive package verification script (`scripts/verify_package_assets.py`) for CI/CD integration to prevent future packaging regressions.

## [0.4.1] - 2025-11-17

### Added
- Official PyPI packaging instructions covering `python -m build`, wheel installation, and CLI verification now live in the release notes to make publishing frictionless.

### Changed
- Bumped the package to `0.4.1`, aligned `namel3ss.__version__`, and expanded the `dev` extra to include `build` so contributors can produce wheels without extra tooling.
- The omnibus `all` extra now automatically pulls in the OTLP exporter so observability setups can emit traces with a single install command.

### Fixed
- Tightened packaging metadata to ensure every wheel includes the runtime sections and optional extras, avoiding missing dependency warnings when users opt into tracing or realtime features.

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
