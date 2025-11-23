# Namel3ss Documentation Style Guide

## Introduction
This guide defines the writing, terminology, and structure standards for all Namel3ss (N3) documentation. It targets experienced software engineers building production systems. Apply these rules to tutorials, language reference, CLI docs, architecture deep-dives, migration guides, comparisons, deployment guides, and examples.

## Tone and Voice
- Write for professionals; assume familiarity with programming, HTTP, CI/CD, and deployment.
- Use precise, neutral language. Avoid hype, sales copy, and unqualified superlatives.
- Prefer short, complete sentences. Explain trade-offs and guarantees.
- When comparing with other frameworks, describe architectural differences rather than value judgments.

> **Warning**  
> Do not use marketing adjectives such as “magical,” “super easy,” or “revolutionary.” Favor accuracy over enthusiasm.

## Terminology and Glossary Usage
- Use the canonical terms defined in `docs/reference/GLOSSARY.md`. Link the first occurrence of each term in a document to the glossary (e.g., `[N3](../reference/GLOSSARY.md#n3)`).
- Do not invent synonyms for glossary entries. Example: use “Agent” (not chatbot), “Application” (not project), “Compilation” (not build step).
- If a new core term is introduced, add it to the glossary in alphabetical order and link it.

## Structure
Every document should contain:
- An intro paragraph summarizing scope and audience.
- Clear section headings.
- Optional callouts for best practices, warnings, or notes using the following format:

> **Best Practice**  
> Use environment variables for all provider keys; do not embed secrets in `.n3` files or source code.

> **Warning**  
> Generated runtime code is not guaranteed to remain stable across major versions.

## Examples
- Use realistic domains (e.g., “User Profile Service,” “Support Assistant,” “Inventory Management”).
- Avoid placeholder names like Foo/Bar/Baz unless demonstrating syntax edge cases.
- Use production conventions: `.env` for secrets, versioned configuration, structured logs, real HTTP verbs and entities.
- Show full command lines for CLI usage and realistic payloads for HTTP examples.

## Comparisons (LangChain, Jac, API frameworks)
- Focus on architecture and execution model:
  - Declarative language (N3) vs imperative orchestration (library frameworks).
  - Generated runtime (backend + frontend + configuration) vs userland assembly.
  - AI-first semantics and lifecycle (compilation to runtime) vs ad-hoc pipeline composition.
- Avoid subjective claims (no “better/worse” or “more powerful”). Discuss trade-offs and deployment implications instead.

## Formatting and Links
- Use Markdown headings consistently (`##`, `###`). Avoid over-nesting.
- Link glossary terms on first use. For hover text, include brief title text if your renderer supports it: `[N3](../reference/GLOSSARY.md#n3 "N3 language")`.
- Keep line length reasonable; wrap prose around 100–120 chars for readability.

## Release Notes and Migration Guides
- Summarize changes succinctly. Include breaking changes, mitigations, and upgrade steps.
- Reference glossary terms and link to relevant sections (e.g., “Compilation,” “Runtime,” “Provider”).

## Review Checklist
- [ ] Tone is professional and neutral; no hype.
- [ ] Glossary terms are linked on first use and used consistently.
- [ ] Sections have clear headings and an intro paragraph.
- [ ] Examples are realistic and production-oriented.
- [ ] Comparisons describe architecture and trade-offs, not superiority.
- [ ] Secrets are handled via environment variables; no inline secrets.
