# CLI Refactoring Implementation Strategy

**Status:** In Progress  
**Started:** November 20, 2025  
**Module:** `namel3ss/cli.py` (1874 lines) → `namel3ss/cli/` package

## Objectives

1. **Modularize** 1874-line CLI file into focused sub-modules (~150-300 lines each)
2. **Centralize** error handling with production-grade exception hierarchy  
3. **Standardize** validation logic across all commands
4. **Document** every public function with comprehensive docstrings
5. **Maintain** 100% backward compatibility for external consumers

## Current Analysis (cli.py structure)

### Commands (11 total)
- `cmd_build` (L730-817) - Build static sites or backends
- `cmd_run` (L820-1006) - Development server with hot reload
- `cmd_test` (L1009-1014) - Run test suite
- `cmd_lint` (L1017-1022) - Lint N3 code
- `cmd_typecheck` (L1025-1030) - Type checking
- `cmd_lsp` (L1033-1055) - Language server protocol
- `cmd_eval` (L1058-1105) - Evaluate chains/prompts
- `cmd_eval_suite` (L1108-1271) - Run evaluation suites  
- `cmd_train` (L1274-1379) - Train ML models
- `cmd_deploy` (L1382-1436) - Deploy applications
- `cmd_doctor` (L1439-1459) - Health check and diagnostics

### Support Functions (~35 functions)
- Context management: `CLIContext`, `_cli_context`, `_create_ephemeral_app_config`
- Error formatting: `_format_cli_error`, `_format_error_detail`, `_traceback_excerpt`
- Environment: `_runtime_env`, `_load_env_file`, `_apply_env_overrides`, `_resolve_env_reference`
- Validation: `_match_app_config`, `_as_path_string`, `_as_string`, `_bool_from_flag`
- Program loading: `_load_cli_app`, `_resolve_program`, `_program_root_for`
- Utilities: `_plural`, `_slugify_model_name`, `_find_first_n3_file`
- Runtime: `prepare_backend`, `run_dev_server`, `check_uvicorn_available`
- Output formatting: `_print_prediction_response_text`, `_print_experiment_result_text`
- Finding: `_find_chain`, `_find_experiment`, `_resolve_model_spec`
- Module loading: `_load_runtime_module`, `_clear_generated_module_cache`

### Entry Point
- `main(argv)` (L1462-end) - Argument parsing and command dispatch

## Target Package Structure

```
namel3ss/cli/
├── __init__.py                    # Public API - main() entry point
├── context.py                     # CLIContext, environment management
├── errors.py                      # Exception hierarchy, error formatting
├── validation.py                  # Argument and config validation
├── utils.py                       # General utilities (plural, slugify, etc.)
├── loading.py                     # Program and module loading
├── output.py                      # Response formatting and printing
├── commands/
│   ├── __init__.py
│   ├── build.py                   # cmd_build + helpers
│   ├── run.py                     # cmd_run + dev server
│   ├── eval.py                    # cmd_eval + cmd_eval_suite
│   ├── train.py                   # cmd_train
│   ├── deploy.py                  # cmd_deploy
│   ├── doctor.py                  # cmd_doctor
│   └── tools.py                   # test, lint, typecheck, lsp
└── devserver.py                   # (May stay as separate module or move here)
```

## Error Hierarchy (cli/errors.py)

```python
class CLIError(Exception):
    """Base error for all CLI operations."""
    def __init__(self, message: str, *, code: str, hint: Optional[str] = None):
        self.message = message
        self.code = code
        self.hint = hint

class CLIConfigError(CLIError):
    """Configuration file or workspace setup errors."""

class CLIValidationError(CLIError):
    """Invalid command arguments or options."""

class CLIRuntimeError(CLIError):
    """Errors during command execution."""

class CLIBuildError(CLIRuntimeError):
    """Build command failures."""

class CLIServerError(CLIRuntimeError):
    """Development server failures."""

# Formatter for production-grade error output
def format_cli_error(exc: BaseException, *, verbose: bool = False) -> str:
    """Format exception for CLI display with context and hints."""
```

## Implementation Phases

### Phase 1: Foundation (Tasks 1-3) - Day 1

**Create core infrastructure modules:**

1. **cli/errors.py**
   - Exception hierarchy (CLIError subclasses)
   - Error formatting with colors and hints
   - Traceback filtering for user-friendly output
   
2. **cli/context.py**
   - CLIContext dataclass
   - Environment resolution (_runtime_env, _load_env_file, _apply_env_overrides)
   - Config matching (_match_app_config, _create_ephemeral_app_config)
   - Effective config resolution (_effective_realtime)
   
3. **cli/validation.py**
   - Type coercion (_as_path_string, _as_string, _bool_from_flag)
   - Environment validation (_resolve_env_reference)
   - Argument normalization (_normalize_run_command_args)

### Phase 2: Utilities & Loading (Tasks 2, continuing) - Day 1-2

4. **cli/utils.py**
   - String utilities (_plural, _slugify_model_name)
   - File discovery (_find_first_n3_file)
   - Finding helpers (_find_chain, _find_experiment, _resolve_model_spec)
   
5. **cli/loading.py**
   - Program resolution (_resolve_program, _program_root_for)
   - App loading (_load_cli_app)
   - Runtime module loading (_load_runtime_module, _clear_generated_module_cache)
   - JSON argument loading (_load_json_argument)
   
6. **cli/output.py**
   - Response printers (_print_prediction_response_text, _print_experiment_result_text)
   - Backend summary (_backend_summary_lines)

### Phase 3: Command Modules (Tasks 4-7) - Day 2-3

7. **cli/commands/build.py**
   - BuildInvocation dataclass
   - cmd_build function
   - Helpers: prepare_backend (may move to separate module)
   - _invoke_tool helper
   
8. **cli/commands/run.py**
   - RunInvocation dataclass
   - cmd_run function
   - Helpers: run_dev_server, check_uvicorn_available
   
9. **cli/commands/eval.py**
   - cmd_eval function
   - cmd_eval_suite function
   - Evaluation helpers
   
10. **cli/commands/train.py**
    - cmd_train function
    - Training-specific helpers
    
11. **cli/commands/deploy.py**
    - cmd_deploy function
    - Deployment helpers
    
12. **cli/commands/doctor.py**
    - cmd_doctor function
    - Health check logic
    
13. **cli/commands/tools.py**
    - cmd_test, cmd_lint, cmd_typecheck, cmd_lsp
    - Tool invocation helpers

### Phase 4: Main Entry Point (Task 9) - Day 3

14. **cli/__init__.py**
    - main() function (cleaned up)
    - Argument parser setup
    - Command dispatch
    - Public API exports

### Phase 5: Testing & Validation (Task 10) - Day 3-4

15. **Test each command in isolation**
16. **Test error handling paths**
17. **Test backward compatibility**
18. **Update any broken imports in rest of codebase**

## Implementation Rules

### Error Handling
- **Every command** must catch and wrap exceptions as CLIError subclasses
- **No bare `except Exception`** without re-raising as domain error
- **Always provide context:** file, line, command, operation
- **Include hints:** Actionable suggestions for users

### Validation
- **All CLI arguments** validated through cli/validation.py helpers
- **No inline validation** - use centralized functions
- **Return structured errors** - not booleans

### Docstrings
- **Every public function:** Comprehensive Google-style docstrings
- **Parameters:** Type and purpose documented
- **Returns:** Type and meaning documented  
- **Raises:** All exception types documented
- **Examples:** For complex commands

### Backward Compatibility
- **cli/__init__.py** must export `main()` for existing entry points
- **No breaking changes** to command-line interface
- **Preserve all command flags** and behaviors

## Success Criteria

- ✅ No single file exceeds 500 lines
- ✅ All commands have focused, testable modules
- ✅ Centralized error handling with rich context
- ✅ Centralized validation logic
- ✅ 100% command coverage with docstrings
- ✅ All tests pass
- ✅ `n3 --help` works identically
- ✅ All existing commands work identically

## Migration Checklist

- [x] Create folder structure
- [ ] Implement cli/errors.py with exception hierarchy
- [ ] Implement cli/context.py with environment management
- [ ] Implement cli/validation.py with validators
- [ ] Implement cli/utils.py with utilities
- [ ] Implement cli/loading.py with loaders
- [ ] Implement cli/output.py with formatters
- [ ] Implement cli/commands/build.py
- [ ] Implement cli/commands/run.py
- [ ] Implement cli/commands/eval.py
- [ ] Implement cli/commands/train.py
- [ ] Implement cli/commands/deploy.py
- [ ] Implement cli/commands/doctor.py
- [ ] Implement cli/commands/tools.py
- [ ] Update cli/__init__.py with main()
- [ ] Test all commands
- [ ] Update imports in codebase
- [ ] Remove old cli.py (or keep as deprecated wrapper)

## Next Steps

1. Start with **cli/errors.py** - foundation for all error handling
2. Move to **cli/context.py** - central to all commands
3. Then **cli/validation.py** - eliminates inline validation
4. Build out command modules one by one
5. Update main() to use new structure
6. Test extensively before committing

---

**This refactoring will transform the CLI from a 1874-line monolith into a clean, maintainable, production-grade command-line interface with proper separation of concerns.**
