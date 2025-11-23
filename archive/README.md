# Archive Directory

This directory contains legacy files, build outputs, and historical content that was moved from the repository root to maintain a clean project structure.

## Structure

```
archive/
├── legacy-outputs/     # Build outputs, test results, and temporary files
└── legacy-examples/    # Old example applications and configurations
```

## Legacy Outputs (`legacy-outputs/`)

Contains various build outputs and test results:

- `test_*_output/` - Test execution outputs
- `build_test*/` - Build testing directories  
- `examples_backup/` - Backup of old example structure
- `support_agent_output/` - Agent support system outputs
- `rlhf_storage/` - Reinforcement Learning from Human Feedback storage
- `test_results.txt` - Historical test results
- `conftest.py` - Old pytest configuration
- `*.html` - Test integration reports

## Legacy Examples (`legacy-examples/`)

Backup of previous example applications and configurations that have been superseded by the current standardized example structure.

## Purpose

These files are preserved for historical reference and potential recovery needs, but are not part of the active development workflow. The current clean repository structure focuses on:

- `examples/` - Current standardized examples
- `tests/` - Active test suite  
- `scripts/` - Organized utility scripts
- `docs/` - Structured documentation

## Maintenance

Periodically review archived content for:
- Files that can be safely deleted
- Content that should be migrated back to active use
- Historical information needed for documentation

## Note

If you need to reference or restore any archived content, check this directory first before assuming it was lost during the repository cleanup.