"""
Nameless (N3) programming language package.

This package contains the core components required to parse a Namel3ss
(`.n3`) file and generate a working application.  The language is a
human‑friendly, declarative language designed for rapidly building
internal tools, dashboards and full‑stack applications.  By writing in
simple English‑like syntax, developers describe the structure of an
application (pages, datasets, tables, charts, actions, themes, etc.)
and the compiler translates that description into concrete Python and
JavaScript artefacts.

The code is organised into several modules:

* ``ast`` – dataclasses representing the abstract syntax tree used by
  the compiler.  These classes are the canonical in‑memory
  representation of an N3 program.
* ``parser`` – a hand written parser that reads an N3 file and
  produces an AST.  It handles indentation, quotes and simple
  statements along with nested blocks.
* ``codegen`` – subpackage responsible for turning the AST into
  concrete backend and frontend source code.  The current
  implementation generates static HTML/JS files using minimal
  dependencies (Chart.js via CDN) to make it easy to run without
  extra infrastructure.
* ``cli`` – a command line interface that ties everything together.
  It allows you to parse a file, preview the AST and build the
  generated site.  Future enhancements may include running a live
  development server, exporting React/FastAPI projects or deploying
  directly to the cloud.

The goal of the initial implementation is to illustrate the
feasibility of the Namel3ss language and provide a working
proof‑of‑concept.  As such, many features are simplified.  For
example, database connections are not yet implemented and data from
tables is not automatically pulled into tables or charts.  However,
the scaffolding is intentionally laid out to make future
enhancements—such as dynamic data loading, forms, authentication and
inline Python/React blocks—straightforward to add.
"""
