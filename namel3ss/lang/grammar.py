"""Grammar-driven parser entry points for Namel3ss `.n3` programs.

This module introduces a lightweight EBNF description of the supported
surface syntax and a reference implementation that produces the existing
AST structures used throughout the runtime.  The grammar intentionally
captures the high-level layout of a module while delegating certain
expression details to the established expression parser mixin.

```
module         ::= directive* declaration*
directive      ::= module_decl | import_stmt | language_version
module_decl    ::= "module" dotted_name NEWLINE
import_stmt    ::= "import" dotted_name ("as" NAME)? (":" import_target ("," import_target)*)? NEWLINE
import_target  ::= NAME ("as" NAME)?
language_version ::= "language_version" STRING NEWLINE

declaration    ::= app_decl | theme_block | dataset_def | frame_def | page_block
                 | model_block | ai_model_block | prompt_block | memory_block
                 | template_block | chain_block | experiment_block | crud_block
                 | evaluator_block | metric_block | guardrail_block

app_decl       ::= "app" STRING ("connects to" NAME STRING)? "."?
theme_block    ::= "theme" ":" INDENT theme_entry+ DEDENT

dataset_def    ::= "dataset" STRING "from" source_ref ":" INDENT dataset_stmt+ DEDENT
dataset_stmt   ::= "filter by" ":" expression
                 | "group by" ":" name_list
                 | "order by" ":" name_list
                 | "transform" NAME ":" expression

frame_def      ::= "frame" STRING ("from" source_ref)? ":" INDENT frame_stmt+ DEDENT
frame_stmt     ::= "columns" ":" column_list
                 | "description" ":" STRING

page_block     ::= "page" STRING "at" STRING ":" INDENT page_stmt+ DEDENT
page_stmt      ::= show_stmt | form_stmt | action_stmt | control_flow_stmt
control_flow_stmt ::= if_stmt | for_stmt
if_stmt        ::= "if" expression ":" INDENT page_stmt+ DEDENT (elif_stmt)* (else_stmt)?
for_stmt       ::= "for" NAME "in" ("dataset"|"table"|"frame") NAME ":" INDENT page_stmt+ DEDENT
show_stmt      ::= "show" ("text"|"table"|"chart") ...
```

Only a subset of the listed productions are implemented at this stage;
the focus of Phase 1 is to support realistic modules that declare an app,
datasets, frames, pages, and page-level control-flow.  The grammar text
serves both as documentation and as a roadmap for future incremental
coverage.

----

REFACTORING NOTE: This module has been refactored into a modular package.
Original: 1,993 lines, monolithic _GrammarModuleParser class
New structure: 14 focused modules in namel3ss/lang/grammar/ package
Total: 2,086 lines (+4.7% overhead for modularity)

Modules:
  - constants.py: Regex patterns (39 lines)
  - helpers.py: Helper classes (12 lines)
  - directives.py: Module/import parsing (78 lines)
  - declarations.py: App/theme/dataset/frame (172 lines)
  - pages.py: Page statements (166 lines)
  - ai_components.py: LLM/tool parsing (237 lines)
  - prompts.py: Structured prompts (389 lines)
  - rag.py: RAG parsing (151 lines)
  - agents.py: Agent/graph (154 lines)
  - policy.py: Policy parsing (93 lines)
  - utility_parsers.py: Utility parsers (224 lines)
  - functions.py: Function/rule defs (64 lines)
  - utility_methods.py: Core utilities (74 lines)
  - parser.py: Main composition (213 lines)
"""

from __future__ import annotations

from typing import Optional

from namel3ss.ast.program import Module

# Re-export everything from the grammar package for backward compatibility
from namel3ss.lang.grammar import (
    parse_module,
    GrammarUnsupportedError,
    _GrammarModuleParser
)

__all__ = ["parse_module", "GrammarUnsupportedError", "_GrammarModuleParser"]
