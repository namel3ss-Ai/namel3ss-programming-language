"""
Feature → Dependency Mapping Specification

This is the single source of truth for mapping Namel3ss features to their
required Python and NPM dependencies. It's designed to be:

1. Extensible: Easy to add new mappings
2. Maintainable: Clear categorization and documentation
3. Testable: Can be validated programmatically
4. Version-aware: Supports version constraints

Architecture:
------------
Each feature (e.g., "openai", "chat", "file_upload") maps to a set of
dependencies with version constraints. Dependencies are categorized by
runtime (backend/frontend) and feature area.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class FeatureCategory(str, Enum):
    """Categories of features for organization"""
    CORE = "core"
    AI_PROVIDER = "ai_provider"
    DATABASE = "database"
    REALTIME = "realtime"
    UI_COMPONENT = "ui_component"
    OBSERVABILITY = "observability"
    SECURITY = "security"
    FILE_STORAGE = "file_storage"
    CACHE = "cache"


@dataclass(frozen=True)
class PythonPackage:
    """Python package specification with version constraint"""
    name: str
    version: str = ""  # Empty means latest, e.g. ">=1.0,<2.0"
    optional: bool = False
    extras: tuple = field(default_factory=tuple)
    
    def to_requirement(self) -> str:
        """Generate requirements.txt line"""
        base = self.name
        if self.extras:
            base += f"[{','.join(self.extras)}]"
        if self.version:
            base += self.version
        return base


@dataclass(frozen=True)
class NPMPackage:
    """NPM package specification with version constraint"""
    name: str
    version: str = "^1.0.0"  # Default to caret range
    dev: bool = False
    
    def to_package_json_entry(self) -> tuple[str, str]:
        """Generate package.json entry"""
        return (self.name, self.version)


@dataclass
class DependencySpec:
    """Complete dependency specification for a feature"""
    feature_id: str
    category: FeatureCategory
    description: str
    python_packages: List[PythonPackage] = field(default_factory=list)
    npm_packages: List[NPMPackage] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)  # Other features that must also be enabled


# =============================================================================
# CORE DEPENDENCIES (Always Required)
# =============================================================================

CORE_BACKEND_DEPS = [
    PythonPackage("fastapi", ">=0.110,<1.0"),
    PythonPackage("uvicorn", ">=0.30,<0.31"),
    PythonPackage("pydantic", ">=2.7,<3.0"),
    PythonPackage("httpx", ">=0.28,<0.29"),
]

CORE_FRONTEND_DEPS = [
    NPMPackage("react", "^18.2.0"),
    NPMPackage("react-dom", "^18.2.0"),
    NPMPackage("@types/react", "^18.2.0", dev=True),
    NPMPackage("@types/react-dom", "^18.2.0", dev=True),
    NPMPackage("typescript", "^5.0.0", dev=True),
    NPMPackage("vite", "^5.0.0", dev=True),
    NPMPackage("@vitejs/plugin-react", "^4.2.0", dev=True),
]


# =============================================================================
# DEPENDENCY SPECIFICATIONS BY FEATURE
# =============================================================================

DEPENDENCY_SPECS: Dict[str, DependencySpec] = {
    
    # =========================================================================
    # Core Features
    # =========================================================================
    
    "core": DependencySpec(
        feature_id="core",
        category=FeatureCategory.CORE,
        description="Core framework dependencies (always required)",
        python_packages=CORE_BACKEND_DEPS,
        npm_packages=CORE_FRONTEND_DEPS,
    ),
    
    "sql": DependencySpec(
        feature_id="sql",
        category=FeatureCategory.DATABASE,
        description="SQL database support via SQLAlchemy",
        python_packages=[
            PythonPackage("sqlalchemy", ">=2.0,<3.0"),
            PythonPackage("asyncpg", ">=0.29,<0.30"),
            PythonPackage("psycopg", ">=3.2,<4.0", extras=["binary"]),
        ],
    ),
    
    "postgres": DependencySpec(
        feature_id="postgres",
        category=FeatureCategory.DATABASE,
        description="PostgreSQL database support",
        python_packages=[
            PythonPackage("sqlalchemy", ">=2.0,<3.0"),
            PythonPackage("asyncpg", ">=0.29,<0.30"),
            PythonPackage("psycopg", ">=3.2,<4.0", extras=["binary"]),
        ],
        requires=["sql"],
    ),
    
    "mysql": DependencySpec(
        feature_id="mysql",
        category=FeatureCategory.DATABASE,
        description="MySQL database support",
        python_packages=[
            PythonPackage("sqlalchemy", ">=2.0,<3.0"),
            PythonPackage("aiomysql", ">=0.2,<0.3"),
        ],
        requires=["sql"],
    ),
    
    "mongo": DependencySpec(
        feature_id="mongo",
        category=FeatureCategory.DATABASE,
        description="MongoDB database support",
        python_packages=[
            PythonPackage("motor", ">=3.0,<4.0"),
            PythonPackage("pymongo", ">=4.0,<5.0"),
        ],
    ),
    
    # =========================================================================
    # AI Provider Features
    # =========================================================================
    
    "openai": DependencySpec(
        feature_id="openai",
        category=FeatureCategory.AI_PROVIDER,
        description="OpenAI API integration (GPT models)",
        python_packages=[
            PythonPackage("openai", ">=1.0,<2.0"),
            PythonPackage("tiktoken", ">=0.7,<1.0"),
        ],
    ),
    
    "anthropic": DependencySpec(
        feature_id="anthropic",
        category=FeatureCategory.AI_PROVIDER,
        description="Anthropic API integration (Claude models)",
        python_packages=[
            PythonPackage("anthropic", ">=0.40,<1.0"),
        ],
    ),
    
    "ollama": DependencySpec(
        feature_id="ollama",
        category=FeatureCategory.AI_PROVIDER,
        description="Ollama local model support",
        python_packages=[
            PythonPackage("ollama-python", ">=0.1.0,<1.0"),
            PythonPackage("httpx", ">=0.24.0"),
        ],
    ),
    
    "vllm": DependencySpec(
        feature_id="vllm",
        category=FeatureCategory.AI_PROVIDER,
        description="vLLM inference engine support",
        python_packages=[
            PythonPackage("vllm", ">=0.4.0,<1.0"),
            PythonPackage("httpx", ">=0.24.0"),
        ],
    ),
    
    # =========================================================================
    # UI Component Features
    # =========================================================================
    
    "chat": DependencySpec(
        feature_id="chat",
        category=FeatureCategory.UI_COMPONENT,
        description="Chat interface components",
        npm_packages=[
            NPMPackage("@radix-ui/react-scroll-area", "^1.0.0"),
            NPMPackage("lucide-react", "^0.300.0"),
        ],
    ),
    
    "file_upload": DependencySpec(
        feature_id="file_upload",
        category=FeatureCategory.UI_COMPONENT,
        description="File upload components",
        python_packages=[
            PythonPackage("python-multipart", ">=0.0.6"),
        ],
        npm_packages=[
            NPMPackage("react-dropzone", "^14.2.0"),
        ],
    ),
    
    "chart": DependencySpec(
        feature_id="chart",
        category=FeatureCategory.UI_COMPONENT,
        description="Data visualization charts",
        npm_packages=[
            NPMPackage("recharts", "^2.10.0"),
            NPMPackage("d3", "^7.8.0"),
        ],
    ),
    
    "data_table": DependencySpec(
        feature_id="data_table",
        category=FeatureCategory.UI_COMPONENT,
        description="Interactive data tables",
        npm_packages=[
            NPMPackage("@tanstack/react-table", "^8.10.0"),
        ],
    ),
    
    "form": DependencySpec(
        feature_id="form",
        category=FeatureCategory.UI_COMPONENT,
        description="Form components with validation",
        npm_packages=[
            NPMPackage("react-hook-form", "^7.48.0"),
            NPMPackage("zod", "^3.22.0"),
        ],
    ),
    
    "markdown": DependencySpec(
        feature_id="markdown",
        category=FeatureCategory.UI_COMPONENT,
        description="Markdown rendering",
        npm_packages=[
            NPMPackage("react-markdown", "^9.0.0"),
            NPMPackage("remark-gfm", "^4.0.0"),
        ],
    ),
    
    "code_editor": DependencySpec(
        feature_id="code_editor",
        category=FeatureCategory.UI_COMPONENT,
        description="Code editor component",
        npm_packages=[
            NPMPackage("@monaco-editor/react", "^4.6.0"),
        ],
    ),
    
    # =========================================================================
    # Real-time Features
    # =========================================================================
    
    "websockets": DependencySpec(
        feature_id="websockets",
        category=FeatureCategory.REALTIME,
        description="WebSocket support for real-time features",
        python_packages=[
            PythonPackage("websockets", ">=12.0,<13.0"),
        ],
    ),
    
    "redis": DependencySpec(
        feature_id="redis",
        category=FeatureCategory.CACHE,
        description="Redis for caching and pub/sub",
        python_packages=[
            PythonPackage("redis", ">=5.0,<6.0"),
        ],
    ),
    
    # =========================================================================
    # Observability Features
    # =========================================================================
    
    "otel": DependencySpec(
        feature_id="otel",
        category=FeatureCategory.OBSERVABILITY,
        description="OpenTelemetry instrumentation",
        python_packages=[
            PythonPackage("opentelemetry-api", ">=1.25,<2.0"),
            PythonPackage("opentelemetry-sdk", ">=1.25,<2.0"),
            PythonPackage("opentelemetry-instrumentation-fastapi", ">=0.46b0,<1.0"),
            PythonPackage("opentelemetry-exporter-otlp", ">=1.25,<2.0"),
        ],
    ),
    
    # =========================================================================
    # Development Tools
    # =========================================================================
    
    "dev_tools": DependencySpec(
        feature_id="dev_tools",
        category=FeatureCategory.CORE,
        description="Development tooling (linting, testing, etc.)",
        python_packages=[
            PythonPackage("pytest", ">=8.0,<9.0"),
            PythonPackage("pytest-asyncio", ">=0.23,<1.0"),
            PythonPackage("black", ">=24.0,<25.0"),
            PythonPackage("ruff", ">=0.5,<1.0"),
        ],
        npm_packages=[
            NPMPackage("eslint", "^8.0.0", dev=True),
            NPMPackage("prettier", "^3.0.0", dev=True),
            NPMPackage("@typescript-eslint/eslint-plugin", "^6.0.0", dev=True),
            NPMPackage("@typescript-eslint/parser", "^6.0.0", dev=True),
        ],
    ),
}


def get_dependency_spec() -> Dict[str, DependencySpec]:
    """Get the complete dependency specification mapping"""
    return DEPENDENCY_SPECS.copy()


def get_feature_spec(feature_id: str) -> Optional[DependencySpec]:
    """Get the dependency spec for a specific feature"""
    return DEPENDENCY_SPECS.get(feature_id)


def get_features_by_category(category: FeatureCategory) -> List[DependencySpec]:
    """Get all features in a specific category"""
    return [spec for spec in DEPENDENCY_SPECS.values() if spec.category == category]


def get_python_packages_for_features(features: Set[str] | List[str]) -> List[PythonPackage]:
    """Get all Python packages needed for a set of features"""
    # Convert to set if list was provided
    if isinstance(features, list):
        features = set(features)
    
    packages = []
    seen_names = set()
    
    # Always include core
    features_to_process = features | {"core"}
    processed = set()
    
    # Process features (iterate over copy to avoid modification during iteration)
    while features_to_process:
        feature = features_to_process.pop()
        if feature in processed:
            continue
        processed.add(feature)
        
        if feature in DEPENDENCY_SPECS:
            spec = DEPENDENCY_SPECS[feature]
            # Add required dependencies
            if spec.requires:
                new_features = set(spec.requires) - processed
                features_to_process.update(new_features)
            # Add packages
            for pkg in spec.python_packages:
                if pkg.name not in seen_names:
                    packages.append(pkg)
                    seen_names.add(pkg.name)
    
    return packages


def get_npm_packages_for_features(features: Set[str] | List[str]) -> List[NPMPackage]:
    """Get all NPM packages needed for a set of features"""
    # Convert to set if list was provided
    if isinstance(features, list):
        features = set(features)
    
    packages = []
    seen_names = set()
    
    # Always include core
    features_to_process = features | {"core"}
    processed = set()
    
    # Process features (iterate over copy to avoid modification during iteration)
    while features_to_process:
        feature = features_to_process.pop()
        if feature in processed:
            continue
        processed.add(feature)
        
        if feature in DEPENDENCY_SPECS:
            spec = DEPENDENCY_SPECS[feature]
            # Add required dependencies
            if spec.requires:
                new_features = set(spec.requires) - processed
                features_to_process.update(new_features)
            # Add packages
            for pkg in spec.npm_packages:
                if pkg.name not in seen_names:
                    packages.append(pkg)
                    seen_names.add(pkg.name)
    
    return packages
