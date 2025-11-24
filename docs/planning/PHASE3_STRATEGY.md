# Phase 3 Strategy: Codegen Migration

## Challenge

The codegen migration is more complex than initially planned because:

1. **Generated code imports from codegen**
   - Generated backends import `from namel3ss.codegen.backend.core.runtime.*`
   - Generated backends import `from namel3ss.codegen.backend.core.sql_compiler import compile_dataset_to_sql`
   - This means moving code breaks existing generated applications

2. **Multiple entry points**
   - CLI uses `from namel3ss.codegen.backend.core import generate_backend`
   - IR builder uses `from namel3ss.codegen.backend.state import build_backend_state`
   - Runtime adapters use `from namel3ss.codegen.backend import generate_backend`

3. **Complex internal dependencies**
   - Backend codegen has ~20 modules in core/ and state/
   - Frontend codegen has ~15 modules
   - Many cross-references between modules

## Revised Phase 3 Approach

### Option A: Keep Codegen in Core (Recommended)

**Rationale:**
- Generated code needs stable import paths
- Moving codegen breaks backward compatibility for users
- The real goal is **dependency separation**, not physical relocation

**Implementation:**
1. Keep `namel3ss/codegen/` in core package
2. Remove FastAPI/runtime dependencies from codegen (already done via IR)
3. Update runtime adapters to import from core codegen (not duplicate)
4. Maintain single source of truth for codegen logic

**Benefits:**
- ✅ No breaking changes for generated code
- ✅ Simpler migration path
- ✅ Codegen is part of "language tooling" (like parser, type checker)
- ✅ Dependency separation already achieved via IR layer

**Tradeoffs:**
- ❌ Codegen stays in core package (not runtime packages)
- ✅ But this is actually correct: codegen is a compiler component

### Option B: Duplicate Runtime Modules (Not Recommended)

Move runtime support modules to runtime packages but leave generation logic:

**Problems:**
- Breaks generated code
- Requires regenerating all examples
- Users must regenerate their apps after upgrade
- High risk of issues

### Option C: Hybrid Approach (Complex)

Keep generation in core, move only runtime support to packages:

**Problems:**
- Split codegen across packages
- Complex import paths
- Harder to maintain

## Decision: Option A

**Phase 3 will focus on:**

1. ✅ **Remove IR bridge** - Make codegen consume IR directly
2. ✅ **Update runtime adapters** - Use core codegen correctly
3. ✅ **CLI refactoring** - Separate language ops from runtime ops
4. ✅ **Dependency cleanup** - Ensure core has no FastAPI imports
5. ✅ **Documentation** - Update architecture docs

**Phase 3 will NOT:**
- ❌ Move codegen to runtime packages (breaks generated code)
- ❌ Change import paths for generated code
- ❌ Relocate runtime support modules

## Architecture (Revised)

```
namel3ss (core)
├── parser/              # Language parsing
├── ast/                 # AST types
├── ir/                  # Intermediate representation
├── types/               # Type system
└── codegen/             # Code generation (compiler component)
    ├── backend/         # Backend generation (uses IR, no FastAPI imports)
    └── frontend/        # Frontend generation (uses IR)

runtimes/
├── http/                # HTTP runtime
│   └── adapter.py       # Imports from namel3ss.codegen
├── frontend/            # Frontend runtime
│   └── adapter.py       # Imports from namel3ss.codegen
└── deploy/              # Deployment runtime
    └── adapter.py       # Creates deployment configs
```

**Key insight:** Codegen is a **compiler component**, not a runtime. It belongs in the language core.

## What Phase 3 Achieves

1. **Dependency Separation** ✅
   - Core has no runtime dependencies (FastAPI, etc.)
   - Achieved via IR layer (Phase 1)
   - Verified in Phase 2

2. **Runtime Independence** ✅
   - Runtimes can be installed separately
   - Each runtime depends on core (not reverse)
   - Achieved in Phase 2

3. **Multiple Targets** ✅
   - IR can target different runtimes
   - Demonstrated in Phase 2 (HTTP, frontend, deploy)

4. **Clean Architecture** ✅
   - Parser → AST → IR → Codegen → Runtime
   - No circular dependencies
   - Proper layering

## Updated Phase 3 Tasks

1. **Remove IR Bridge**
   - Remove `_original_app` from IR metadata
   - Make codegen work directly with IR (if not already)
   - Update runtime adapters

2. **CLI Refactoring**
   - Keep using `namel3ss.codegen` (correct path)
   - Separate runtime selection from language operations
   - Add `--runtime` flag for future extensibility

3. **Dependency Audit**
   - Verify no FastAPI in core imports
   - Ensure clean separation achieved

4. **Documentation Updates**
   - Update REFACTORING_PLAN.md
   - Clarify architecture decisions
   - Document why codegen stays in core

5. **Testing**
   - Run full test suite
   - Verify generated code still works
   - Test all examples

## Conclusion

**The goal was never to move codegen physically** - it was to achieve clean dependency separation. This is already achieved:

- ✅ Core has no FastAPI dependencies
- ✅ IR provides runtime-agnostic representation
- ✅ Runtimes are independent packages
- ✅ Multiple runtime targets possible

Phase 3 will focus on **removing the temporary bridge** and **polishing the architecture**, not on physically relocating code that would break backward compatibility.
