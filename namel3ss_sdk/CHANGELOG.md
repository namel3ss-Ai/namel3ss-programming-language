# Changelog - Namel3ss SDK

All notable changes to the Namel3ss Python SDK will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

- Nothing yet.

## [0.2.0] - 2025-11-22

### Changed
- **Production Metadata**: Enhanced PyPI metadata with structured README configuration (`text/markdown`), MIT license classifier, and comprehensive Python version classifiers (3.10, 3.11, 3.12).
- **Improved Description**: Updated package description to clarify SDK provides both remote and in-process execution modes.
- **Enhanced Classifiers**: Added FastAPI framework classifier, AI/ML topic classifiers, and explicit MIT license classifier for better PyPI discoverability.
- **Documentation URLs**: Added explicit Issues tracker URL to project metadata.

### Added
- **SDK Changelog**: Introduced dedicated CHANGELOG.md for tracking SDK-specific version history separate from core language package.
- **Release Documentation**: Created comprehensive build and release workflow documentation for independent SDK releases.

### Notes
- This is a MINOR version bump (0.1.0 → 0.2.0) representing packaging and metadata improvements.
- All runtime functionality remains unchanged and backward compatible.
- SDK can now be built and released independently from the core `namel3ss` package.

## [0.1.0] - 2025-11-15

### Added
- **Initial SDK Release**: Production-grade Python SDK for integrating Namel3ss into existing applications.
- **Remote Execution**: HTTP-based client for calling chains, agents, prompts, and RAG pipelines on remote Namel3ss servers.
- **In-Process Execution**: Embedded runtime for running `.ai` workflows directly in Python processes.
- **Fault Tolerance**: Circuit breaker pattern with configurable thresholds and automatic recovery.
- **Retry Logic**: Exponential backoff with jitter for transient failure handling.
- **Async Support**: Full async/await support with context managers and concurrent execution.
- **Type Safety**: Pydantic-based configuration with comprehensive type hints.
- **Observability**: OpenTelemetry instrumentation support (optional telemetry extra).
- **Exception Hierarchy**: Structured error types (N3ClientError, N3TimeoutError, N3AuthError, etc.) with request ID tracking.
- **Zero-Config Defaults**: Automatic configuration from environment variables and `.env` files.
- **Security**: TLS enforcement, token rotation support, no PII/secrets in logs.

### Security
- TLS required by default for remote connections
- Configurable SSL verification
- Token-based authentication with rotation support
- Request ID tracking for audit trails
