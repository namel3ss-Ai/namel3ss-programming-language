# namel3ss Module and Package System

The namel3ss programming language features a comprehensive module and package system designed for building scalable AI applications. This system provides hierarchical organization, dependency management, and seamless integration across projects.

## Table of Contents

1. [Overview](#overview)
2. [Module System](#module-system)
3. [Package System](#package-system)
4. [Workspace Configuration](#workspace-configuration)
5. [CLI Commands](#cli-commands)
6. [Best Practices](#best-practices)
7. [Migration Guide](#migration-guide)
8. [Examples](#examples)

## Overview

The namel3ss module and package system consists of:

- **Modules**: Individual `.ai` files containing prompts, tools, apps, and workflows
- **Packages**: Collections of modules with metadata and dependencies
- **Workspaces**: Root directories containing modules and packages
- **Dependencies**: Version-controlled relationships between packages

### Key Features

- ✅ Hierarchical module naming (`company.core::utils.validation`)
- ✅ Package manifests with dependency resolution
- ✅ Version constraints and conflict detection
- ✅ Workspace-based project organization
- ✅ Cross-package imports and exports
- ✅ CLI tools for package management
- ✅ Production-ready architecture

## Module System

### Basic Module Structure

Every namel3ss module is defined in a `.ai` file:

```namel3ss
module customer_service

prompt "greet_customer" {
    template: "Hello {{customer_name}}, how can I help you today?"
}

tool "ticket_system" {
    type: "database"
    connection: "tickets.db"
}

app "CustomerServiceApp" {
    description: "Customer support application"
}
```

### Hierarchical Module Naming

Modules can be organized hierarchically using dot notation:

```namel3ss
# File: analytics/reports.ai
module analytics.reports

prompt "daily_report" {
    template: "Daily Report for {{date}}: {{metrics}}"
}

# File: analytics/metrics/performance.ai
module analytics.metrics.performance

prompt "measure_latency" {
    template: "Measuring latency: {{response_time}}ms"
}
```

### Use Statements

Import functionality from other modules:

```namel3ss
module main

# Import entire module
use analytics.reports

# Import with alias
use analytics.metrics.performance as perf

# Import from package
use company.core::utils.validation

app "MainApp" {
    description: "Application using multiple modules"
}
```

### Module Exports

Control which components are available for import:

```namel3ss
module utils.helpers

prompt "format_data" {
    template: "Formatted: {{data}}"
}

# Internal prompt (not exported)
prompt "_internal_helper" {
    template: "Internal use only"
}

# Export statement
export {
    "format_data"  # Only this prompt is importable
}
```

## Package System

### Package Structure

A package is a directory containing a `namel3ss.toml` manifest and modules:

```
my_package/
├── namel3ss.toml          # Package manifest
├── main.ai                # Main module
├── utils/
│   ├── validation.ai      # Utils module
│   └── formatting.ai      # Utils module
└── README.md              # Package documentation
```

### Package Manifest (`namel3ss.toml`)

```toml
[package]
name = "company.analytics"
version = "1.5.0"
description = "Analytics and reporting tools for company applications"
authors = ["Company DevTeam <dev@company.com>"]
license = "MIT"
homepage = "https://github.com/company/analytics-package"

[dependencies]
"company.core" = ">=2.0.0,<3.0.0"
"external.charts" = "^1.2.0"

[dev-dependencies]
"testing.framework" = ">=1.0.0"

# Modules included in this package
modules = [
    "reports.daily",
    "reports.monthly",
    "metrics.performance",
    "visualizations"
]

# Control which modules are exported
[module-exports]
"reports" = ["reports.daily", "reports.monthly"]
"metrics" = "metrics.performance"
# visualizations is internal-only

[namel3ss]
version = ">=1.0.0"  # Minimum namel3ss version required
```

### Version Constraints

Supports semantic versioning with various constraint formats:

```toml
[dependencies]
"exact.version" = "==1.2.3"     # Exact version
"minimum" = ">=1.0.0"           # Minimum version
"compatible" = "^1.2.0"         # Compatible (1.2.0 <= v < 2.0.0)
"patch" = "~=1.2.0"             # Patch level (~1.2.0)
"range" = ">=1.0.0,<2.0.0"      # Version range
```

### Creating a Package

1. **Initialize package directory:**
   ```bash
   mkdir my_analytics_package
   cd my_analytics_package
   ```

2. **Create package manifest:**
   ```bash
   # Create namel3ss.toml with package metadata
   ```

3. **Add modules:**
   ```bash
   # Create .ai files with your modules
   ```

4. **Test package:**
   ```bash
   namel3ss packages check
   ```

## Workspace Configuration

### Workspace Structure

A workspace is the root directory for your project:

```
my_project/
├── namel3ss.toml          # Workspace config
├── main.ai                # Workspace modules
├── src/
│   └── components.ai
├── packages/              # Local packages
│   └── shared_utils/
└── external/              # External packages
    └── third_party/
```

### Workspace Configuration (`namel3ss.toml`)

```toml
[workspace]
name = "my-ai-project"
description = "Advanced AI application workspace"
version = "1.0.0"

# Directories to scan for modules
module_paths = [
    "src",
    "apps",
    "tools"
]

# Directories to scan for packages
package_paths = [
    "packages",
    "external",
    "libs"
]

[dependencies]
# Workspace-level dependencies
"company.core" = ">=2.0.0"

[dev-dependencies]
"testing.utils" = ">=1.0.0"

[workspace.settings]
default_llm_provider = "openai"
enable_type_checking = true
module_discovery_timeout = 30
```

### Workspace Discovery

The namel3ss loader automatically discovers:

1. **Workspace root**: Directory containing `namel3ss.toml`
2. **Modules**: `.ai` files in `module_paths` directories
3. **Packages**: Subdirectories with `namel3ss.toml` in `package_paths`
4. **Dependencies**: Resolved according to version constraints

## CLI Commands

### Package Commands

```bash
# List all packages in workspace
namel3ss packages list
namel3ss packages list --json  # JSON output

# Get package information
namel3ss packages info company.core
namel3ss packages info company.core --json

# Show package dependencies
namel3ss packages deps company.analytics
namel3ss packages deps company.analytics --resolved

# Check dependency consistency
namel3ss packages check
namel3ss packages check --verbose
```

### Module Commands

```bash
# List all modules
namel3ss modules list
namel3ss modules list --package company.core  # Filter by package

# Get module information
namel3ss modules info company.core::utils.validation
namel3ss modules info main --json

# Show module dependencies
namel3ss modules deps main
namel3ss modules deps company.analytics::reports.daily

# Generate dependency graph
namel3ss modules graph
namel3ss modules graph --format dot  # Graphviz format
```

### Example CLI Output

```bash
$ namel3ss packages list
┌─────────────────────┬─────────┬──────────────────────────────┐
│ Package             │ Version │ Description                  │
├─────────────────────┼─────────┼──────────────────────────────┤
│ company.core        │ 2.1.0   │ Core utilities               │
│ company.analytics   │ 1.5.0   │ Analytics and reporting      │
│ external.charts     │ 1.2.3   │ Chart generation library     │
└─────────────────────┴─────────┴──────────────────────────────┘

$ namel3ss modules deps main
Dependencies for module 'main':
├── company.core::utils.validation
├── company.analytics::reports.daily
│   └── company.core::utils.formatting
└── external.charts::bar_chart
```

## Best Practices

### Package Organization

1. **Clear naming**: Use descriptive, hierarchical package names
   ```toml
   name = "company.feature.subfeature"  # Good
   name = "pkg1"                        # Bad
   ```

2. **Semantic versioning**: Follow semver for version numbers
   ```toml
   version = "1.2.3"  # MAJOR.MINOR.PATCH
   ```

3. **Minimal dependencies**: Only depend on packages you actually use
   ```toml
   [dependencies]
   "actually.used" = ">=1.0.0"
   # Don't include unused dependencies
   ```

### Module Design

1. **Single responsibility**: Each module should have a focused purpose
   ```namel3ss
   # Good: Focused module
   module customer.authentication
   
   # Bad: Mixed concerns
   module everything
   ```

2. **Clear exports**: Explicitly control what's importable
   ```namel3ss
   export {
       "public_prompt",
       "public_tool"
       # Don't export internal helpers
   }
   ```

3. **Meaningful names**: Use descriptive names for components
   ```namel3ss
   prompt "authenticate_user" {     # Good
       template: "Login: {{user}}"
   }
   
   prompt "p1" {                   # Bad
       template: "Login: {{user}}"
   }
   ```

### Dependency Management

1. **Version constraints**: Use appropriate constraint levels
   ```toml
   [dependencies]
   "stable.api" = "^2.0.0"      # Compatible updates
   "experimental" = "==1.0.0"   # Pin exact version
   "utilities" = ">=1.0.0"      # Minimum version
   ```

2. **Avoid circular dependencies**: Design packages to have clear hierarchy
   ```
   core ← utils ← analytics    # Good
   core ↔ utils               # Bad (circular)
   ```

3. **Regular updates**: Keep dependencies current
   ```bash
   namel3ss packages check  # Regular dependency checks
   ```

### Workspace Organization

1. **Logical structure**: Organize code by functionality
   ```
   workspace/
   ├── src/core/           # Core functionality
   ├── src/features/       # Feature modules
   ├── apps/              # Applications
   └── packages/          # Local packages
   ```

2. **Clear configuration**: Document workspace settings
   ```toml
   [workspace]
   # Clear description of workspace purpose
   description = "Customer service AI application"
   ```

## Migration Guide

### From Basic Modules to Packages

**Before (basic modules):**
```namel3ss
# File: utils.ai
module utils

prompt "helper" {
    template: "Help: {{text}}"
}

# File: main.ai
module main

use utils

app "MyApp" {
    description: "Simple app"
}
```

**After (packaged modules):**

1. **Create package structure:**
   ```
   utils_package/
   ├── namel3ss.toml
   └── utils.ai
   ```

2. **Add package manifest:**
   ```toml
   [package]
   name = "mycompany.utils"
   version = "1.0.0"
   modules = ["utils"]
   ```

3. **Update imports:**
   ```namel3ss
   # File: main.ai
   module main

   use mycompany.utils::utils

   app "MyApp" {
       description: "Packaged app"
   }
   ```

### From Flat Structure to Hierarchical

**Before:**
```
project/
├── customer_auth.ai
├── customer_profile.ai
├── admin_auth.ai
└── admin_dashboard.ai
```

**After:**
```
project/
├── namel3ss.toml
├── customer/
│   ├── auth.ai          # module customer.auth
│   └── profile.ai       # module customer.profile
└── admin/
    ├── auth.ai          # module admin.auth
    └── dashboard.ai     # module admin.dashboard
```

### Adding Dependencies

**Step 1: Identify dependencies**
```bash
namel3ss modules deps main  # See what main imports
```

**Step 2: Create package manifest**
```toml
[package]
name = "myapp.main"
version = "1.0.0"

[dependencies]
"external.library" = ">=1.0.0"
```

**Step 3: Update imports**
```namel3ss
use external.library::component
```

## Examples

### Example 1: Customer Service Package

**Package structure:**
```
customer_service_package/
├── namel3ss.toml
├── authentication.ai
├── support/
│   ├── tickets.ai
│   └── chat.ai
└── analytics/
    └── reports.ai
```

**Package manifest:**
```toml
[package]
name = "company.customer_service"
version = "2.0.0"
description = "Customer service tools and workflows"

[dependencies]
"company.core" = ">=1.5.0"
"external.chat_api" = "^2.1.0"

modules = [
    "authentication",
    "support.tickets",
    "support.chat",
    "analytics.reports"
]

[module-exports]
"auth" = "authentication"
"support" = ["support.tickets", "support.chat"]
```

**Module example:**
```namel3ss
# File: authentication.ai
module authentication

use company.core::utils.validation

prompt "login_user" {
    template: "Authenticate user: {{username}}"
    validation: {
        username: "required|string|min:3"
    }
}

tool "auth_database" {
    type: "database"
    connection: "auth.db"
    table: "users"
}

workflow "user_login" {
    steps: [
        "validate_credentials",
        "check_permissions",
        "create_session"
    ]
}
```

### Example 2: Analytics Dashboard

**Workspace configuration:**
```toml
[workspace]
name = "analytics-dashboard"
description = "AI-powered analytics dashboard"

module_paths = ["src", "apps"]
package_paths = ["packages", "vendor"]

[dependencies]
"company.analytics" = ">=1.5.0"
"external.charts" = "^2.0.0"
```

**Main application:**
```namel3ss
# File: apps/dashboard.ai
module apps.dashboard

use company.analytics::reports.daily
use company.analytics::metrics.performance as metrics
use external.charts::bar_chart

app "AnalyticsDashboard" {
    name: "AI Analytics Dashboard"
    description: "Real-time analytics with AI insights"
    version: "1.0.0"
}

llm "analytics_ai" {
    provider: "openai"
    model: "gpt-4"
    temperature: 0.2
}

prompt "analyze_metrics" {
    template: """
    Analyze the following metrics and provide insights:
    
    Daily Active Users: {{dau}}
    Conversion Rate: {{conversion_rate}}
    Revenue: {{revenue}}
    
    Please provide:
    1. Key trends
    2. Anomalies or concerns
    3. Recommendations
    """
}

workflow "generate_daily_insights" {
    steps: [
        "fetch_metrics",
        "analyze_metrics",
        "generate_visualizations",
        "create_summary_report"
    ]
}
```

### Example 3: Multi-Package Workspace

**Workspace structure:**
```
ai_platform/
├── namel3ss.toml              # Workspace config
├── apps/
│   ├── web_interface.ai       # Main web app
│   └── api_server.ai          # API server
├── packages/
│   ├── core_utils/            # Core utilities package
│   ├── ml_models/             # ML models package
│   └── data_processing/       # Data processing package
└── vendor/
    └── external_apis/         # External API integrations
```

**Complex dependency graph:**
```namel3ss
# File: apps/web_interface.ai
module apps.web_interface

# Import from multiple packages
use platform.core::authentication
use platform.ml_models::text_classifier
use platform.data_processing::pipelines
use external.apis::payment_gateway

app "AIPlatformWeb" {
    name: "AI Platform Web Interface"
    description: "Web interface for AI platform"
    
    components: [
        "authentication",
        "text_classifier", 
        "data_pipelines",
        "payment_integration"
    ]
}

workflow "process_user_input" {
    steps: [
        "authenticate_user",
        "validate_input",
        "classify_text",
        "process_data",
        "generate_response",
        "log_interaction"
    ]
}
```

## Conclusion

The namel3ss module and package system provides a robust foundation for building scalable AI applications. By following the patterns and best practices outlined in this guide, you can create maintainable, reusable, and well-organized AI codebases.

Key benefits:

- 🏗️ **Scalable architecture** for large AI projects
- 🔗 **Clean dependencies** with version management
- 📦 **Reusable packages** across projects
- 🛠️ **Powerful CLI tools** for package management
- 🔍 **Clear organization** with hierarchical modules
- ⚡ **Production ready** with comprehensive testing

For more examples and advanced usage, see the `/examples` directory in the namel3ss repository.