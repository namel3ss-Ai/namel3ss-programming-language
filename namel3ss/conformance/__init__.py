"""
Namel3ss Language Conformance Suite.

This package provides the conformance test infrastructure for Namel3ss Language 1.0,
enabling validation of language-level behavior across multiple implementations.
"""

from namel3ss.conformance.models import (
    ConformanceTestDescriptor,
    TestPhase,
    TestStatus,
    Diagnostic,
    discover_conformance_tests,
)
from namel3ss.conformance.runner import (
    ConformanceRunner,
    ConformanceTestResult,
    TestResult,
)

__version__ = "1.0.0"

__all__ = [
    "ConformanceTestDescriptor",
    "TestPhase",
    "TestStatus",
    "Diagnostic",
    "discover_conformance_tests",
    "ConformanceRunner",
    "ConformanceTestResult",
    "TestResult",
]
