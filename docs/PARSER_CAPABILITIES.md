# Namel3ss Parser Capabilities & Limitations

This document summarizes what UI patterns and features are currently supported by the Namel3ss parser, based on production example development.

**Last Updated:** November 26, 2025 - Parser enhanced with tool/agent/vector_store support, show tabs, and condition: alias
**Parser Version:** Current main branch

---

## ✅ Fully Supported Features

### Top-Level Declarations

#### Tool Definitions
Define external tools/APIs for LLM function calling:
```n3
tool search_docs:
    description: "Search through document collection"
    parameters:
        query:
            type: string
            description: "Search query"
            required: true
        top_k:
            type: integer
            description: "Number of results"
            required: false
```

#### Agent Definitions
Define AI agents with specific capabilities:
```n3
agent rag_assistant:
    llm: gpt-4
    tools: [search_docs, rerank_results]
    goal: "Answer questions using document search"
    system_prompt: "You are a helpful RAG assistant"
    max_turns: 10
    temperature: 0.7
```

#### Vector Store Configuration
Configure vector databases for RAG:
```n3
vector_store main_store:
    type: pgvector
    table: embeddings
    dimension: 1536
    distance_metric: cosine
```

### Chrome Components
All chrome navigation components work correctly:
- `sidebar` with icon-labeled items
- `navbar` with title and action buttons
- `breadcrumbs` for navigation trails
- Chrome must be declared before page content

### Tabs with Chrome
**NEW:** Tabs now work with chrome components using simple syntax:
```n3
page "Dashboard" at "/dashboard":
    sidebar:
        item "Home" at "/" icon "🏠"
    
    navbar:
        title: "Dashboard"
    
    show tabs:
        tab "Overview":
            show card "Items" from dataset items:
                condition: "status == 'active'"
        
        tab "Details":
            show text "Details content"
```

### Card Display Patterns
Complex nested card structures parse successfully:
```n3
show card "Name" from dataset name:
    filter_by: "field == value"
    group_by: field_name
    item:
        type: card
        header:
            title: "{{field}}"
            badges:
                - field: count
                - field: status
        sections:
            - type: info_grid
              columns: 2  # or 3, or 4
              items:
                  - icon: "📊"
                    label: "Label"
                    values:
                        - field: field_name
                          format: "MMMM DD, YYYY"
            - type: text_section
              content: "{{content_field}}"
        actions:
            - label: "Action"
              icon: "🔥"
              action: "action_name"
              params: "{{id}}"
```

**Supported Nesting Levels:** 3-4 levels deep
**Supported Column Counts:** 2, 3, 4 columns in info_grid
**Multiple Items:** Multiple badges, actions, sections all work

### Dataset Declarations
Simple, flat dataset syntax:
```n3
dataset "name" from table table_name
```

**Requirements:**
- Dataset names must be quoted strings
- Use `from table` syntax only
- NO `columns:` or `fields:` blocks

### Indentation
**Required:** 4-space indentation throughout
**Not Supported:** 2-space indentation (parser will reject)

### Dynamic Routing
Path parameters work correctly:
```n3
page "Detail" at "/item/:id"
page "Nested" at "/category/:cat_id/item/:item_id"
```

### Filtering & Grouping
- `filter_by: "expression"` - Filter data ✅
- `condition: "expression"` - **NEW:** Alias for filter_by ✅
- `group_by: field_name` - Group by field ✅
- `sort_by: field_name` - Sort data ✅

### Data Display Components
The following components parse when used with chrome:
- `show text` - Simple text display ✅
- `show card` - Card grids with datasets ✅
- `show tabs` - **NEW:** Tabs with nested content ✅

---

## ❌ Known Limitations (NOW FIXED!)

### ~~Chrome + Complex Components~~ ✅ FIXED
**Previously:** Chrome components did NOT work with `show tabs`
**NOW:** ✅ Chrome + `show tabs` works perfectly with simple syntax!

### ~~Tabs Usage~~ ✅ FIXED
**Previously:** `show tabs` could not be used on pages with chrome components
**NOW:** ✅ Fully supported with intuitive `tab "Name":` syntax

### ~~Advanced RAG Features~~ ✅ FIXED
**Previously:** Tool/agent/vector_store definitions were not parseable
**NOW:** ✅ Full support for:
- Tool definitions with parameters and descriptions
- Agent definitions with LLM, tools, and configuration
- Vector store configurations

### ~~Property Names~~ ✅ PARTIALLY FIXED
**Previously:** `condition:` property not supported
**NOW:** ✅ `condition:` works as an alias for `filter_by:`

---

## 🎯 Best Practices (Updated)

### 1. Start from Working Examples
Use proven examples as templates:
- `examples/ai-customer-support-console.ai`
- `examples/rag-document-assistant.ai`

### 2. Test Incrementally
When building complex UIs:
1. Start with minimal structure (app, datasets, basic pages)
2. Add chrome components
3. Add simple `show text`
4. Add `show card` with basic structure
5. Gradually add nesting (sections, info_grid, etc.)
6. Test parsing after each addition

### 3. Use 4-Space Indentation
Always use 4 spaces for indentation. The parser will reject 2-space indentation.

### 4. Document vs Implement Strategy
For features not supported by parser:
- **Document** advanced configs (tools, agents, RAG setup) in .md files
- **Implement** UI patterns and data display in .ai files
- Keep .ai files focused on parseable syntax
- Provide comprehensive examples in documentation

### 5. Chrome Component Guidelines (Updated)
- ✅ Chrome + `show text` = Works perfectly
- ✅ Chrome + `show card` = Works perfectly
- ✅ Chrome + `show tabs` = **NOW WORKS!** ✨
- Use simple `tab "Name":` syntax for best results

### 6. Tabs Syntax
Two syntaxes supported:

**Simple (Recommended):**
```n3
show tabs:
    tab "Overview":
        show card "Items" from dataset items
    tab "Details":
        show text "Content"
```

**Verbose:**
```n3
layout tabs:
    tabs:
        - id: overview
          label: "Overview"
          content:
              - show card "Items" from dataset items
```

### 7. RAG Application Structure
For production RAG apps, now you can define everything in one file:
```n3
app "My RAG App"

# Define tools
tool search_docs:
    description: "Search documents"
    parameters:
        query:
            type: string
            required: true

# Define agents
agent rag_assistant:
    llm: gpt-4
    tools: [search_docs]
    goal: "Answer questions accurately"

# Configure vector store
vector_store main_store:
    type: pgvector
    dimension: 1536

# Define datasets
dataset "docs" from table documents

# Build UI with tabs
page "Assistant" at "/assistant":
    sidebar:
        item "Home" at "/" icon "🏠"
    
    show tabs:
        tab "Chat":
            show card "Queries" from dataset queries
        tab "Tools":
            show card "Tool Calls" from dataset logs
```

### 6. Nested Structure Limits
- **Safe depth:** 3-4 levels of nesting
- **Tested pattern:** `card → sections → info_grid → items`
- **Multi-column grids:** 2, 3, 4 columns all work
- **Multiple properties:** Multiple badges, actions, sections all work

---

## 🧪 Testing Approach

When creating complex examples:

1. **Parse test first:**
   ```python
   from namel3ss.parser import Parser
   parser = Parser(code)
   module = parser.parse()
   ```

2. **Verify structure:**
   - Check app name
   - Count datasets
   - Count pages
   - Verify page routes

3. **Test incrementally:**
   - Add one feature at a time
   - Parse after each addition
   - Roll back if parse fails
   - Document the limitation

4. **Run test suites:**
   - Create standalone test files (e.g., `test_example_standalone.py`)
   - Test parsing, structure, components
   - Document expected behavior

---

## 📊 Complexity Examples

### Simple (Always Works)
```n3
page "Home" at "/":
    show text "Hello"
```

### Medium (Works with Chrome)
```n3
page "Library" at "/library":
    sidebar:
        item "Home" at "/"
    navbar:
        title: "Library"
    
    show text "Content"
    
    show card "Items" from dataset items:
        item:
            type: card
            header:
                title: "{{name}}"
```

### Complex (Maximum Tested Depth)
```n3
page "Dashboard" at "/dashboard":
    sidebar:
        item "Home" at "/"
    navbar:
        title: "Dashboard"
    
    show card "Items" from dataset items:
        group_by: category
        item:
            type: card
            header:
                title: "{{name}}"
                badges:
                    - field: status
                    - field: count
            sections:
                - type: info_grid
                  columns: 4
                  items:
                      - icon: "📊"
                        label: "Metric"
                        values:
                            - field: value
                              format: "number"
                - type: text_section
                  content: "{{description}}"
            actions:
                - label: "View"
                  icon: "👁️"
                  action: "view"
                - label: "Edit"
                  icon: "✏️"
                  action: "edit"
```

---

## 🔍 Discovery Methodology

These limitations were discovered through:

1. **Iterative Development:** Building production-grade examples (AI Customer Support Console, RAG Document Assistant)
2. **Parse Testing:** Testing each feature addition with parser
3. **Error Analysis:** Analyzing parser error messages
4. **Pattern Extraction:** Identifying what works vs. what doesn't
5. **Documentation:** Recording findings for future developers

**Philosophy:** Build complex examples to stress-test the language and improve it based on real developer needs, not toy demos.

---

## 📚 Related Resources

- [AI Customer Support Console](../examples/ai-customer-support-console.ai) - Working chrome + card example
- [RAG Document Assistant](../examples/rag-document-assistant.ai) - Complex nested UI patterns
- [RAG Implementation Guide](../examples/rag-document-assistant-and-citation-explorer.md) - Full RAG documentation
- [README](../README.md) - Main project documentation
