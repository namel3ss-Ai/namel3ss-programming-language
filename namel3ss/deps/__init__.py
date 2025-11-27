"""
Namel3ss Dependency Management System

This package provides intelligent dependency detection and generation based on
actual feature usage in Namel3ss projects. It analyzes IR to determine which
packages are needed and generates appropriate dependency files.

Core Components:
---------------
- spec.py: Feature → Dependency mapping (source of truth)
- detector.py: IR analysis to detect feature usage
- generator.py: Dependency file generation (requirements.txt, package.json)
- manager.py: High-level API for dependency operations

Usage:
------
    from namel3ss.deps import DependencyManager
    
    manager = DependencyManager()
    manager.sync_project('./my_project')
"""

from .spec import (
    DependencySpec,
    PythonPackage,
    NPMPackage,
    FeatureCategory,
    get_dependency_spec,
    get_feature_spec,
)
from .detector import FeatureDetector, DetectedFeatures
from .generator import DependencyGenerator, GeneratedDeps
from .manager import DependencyManager

__all__ = [
    # Specs
    "DependencySpec",
    "PythonPackage",
    "NPMPackage",
    "FeatureCategory",
    "get_dependency_spec",
    "get_feature_spec",
    # Detection
    "FeatureDetector",
    "DetectedFeatures",
    # Generation
    "DependencyGenerator",
    "GeneratedDeps",
    # Management
    "DependencyManager",
]
