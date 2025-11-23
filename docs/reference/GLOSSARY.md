# Glossary

Authoritative definitions for core Namel3ss (N3) terminology. Use these terms consistently and link to them on first use in each document.

- **Agent**: An executable behavior in N3 with state, tools, and memory. Not a chatbot. Agents run inside the generated runtime.
- **Application**: A complete deployable unit written in N3. Includes backend, frontend, schemas, configuration, and generated tests.
- **Chain**: A sequence of AI or procedural steps executed under explicit runtime rules. Chains are first-class in N3.
- **Compilation**: The transformation from `.ai` source to runtime assets: backend API, frontend, memory schemas, provider bindings, and generated tests.
- **Memory**: Persistent runtime state used in agent reasoning, retrieval, and context management.
- **N3**: The core AI programming language compiled to runtime systems. Defines agents, memory, chains, datasets, providers, and pages.
- **Namel3ss**: The ecosystem (language, CLI, tooling, compilers, runtime) that implements N3.
- **Provider**: A model backend (e.g., OpenAI, Anthropic, local models) bound through configuration, not code changes.
- **RAG**: Retrieval-Augmented Generation integrated into the N3 runtime, not an external library add-on.
- **Runtime**: The execution environment generated from compilation. Includes backend services, frontend assets, configuration, tests, and deployment scaffolding.
- **Tool**: A callable runtime function or external action bound to agents. Tools expose deterministic actions into agent workflows.
