# Namel3ss Parser Limitations - Education Quiz Suite

## Identified Limitations During Complex Feature Implementation

### 1. **Tool Definition Syntax Conflict**
**Issue**: Dual parser system has incompatible syntax expectations
- **Legacy Parser**: Expects `tool name:` with indented properties
- **Modern Parser**: Expects `tool "name" {}` with curly brace blocks
- **Error**: Neither syntax works reliably across both parsers

**Example that FAILS**:
```namel3ss
# Legacy syntax
tool generate_quiz:
    description: "Generate questions"
    parameters:
        topic:
            type: string
            required: true

# Modern syntax  
tool "generate_quiz" {
    description: "Generate questions"
    parameters: {
        topic: { type: "string", required: true }
    }
}
```

**Impact**: Cannot define tools in DSL files
**Workaround**: Document tools in markdown, implement in Python
**Fix Needed**: Unify tool syntax or make modern parser primary

---

### 2. **Agent Definition Syntax Conflict**
**Issue**: Similar to tools - dual parser incompatibility
- **Legacy Parser**: `agent name:` with indented config
- **Modern Parser**: `agent "name" {}` with curly braces

**Example that FAILS**:
```namel3ss
agent quiz_maker:
    llm: gpt-4
    tools: [generate_quiz]
    temperature: 0.7
```

**Error**: `Unexpected token - Expected: string, Found: identifier`
**Impact**: Cannot define AI agents in DSL
**Workaround**: Document agents separately
**Fix Needed**: Standardize agent syntax across parsers

---

### 3. **Show Card Actions/Buttons Not Parsed**
**Issue**: Card item `actions:` property not recognized
- Parser error: "Unknown item property: - label"
- Cannot define button actions within cards

**Example that FAILS**:
```namel3ss
show card "Quizzes" from dataset quizzes:
    item:
        type: card
        header:
            title: "{{title}}"
        actions:
            - label: "Edit"
              icon: "✏️"
              type: primary
```

**Error**: `Unknown item property: - label` at line with `- label:`
**Impact**: Cards cannot have interactive buttons
**Workaround**: Use simple text-only cards
**Fix Needed**: Implement actions parsing in `_parse_card_item_config`

---

### 4. **Chart Components Not Supported in Simple Pages**
**Issue**: `chart_type`, `x_axis`, `y_axis` properties not recognized
- Works in complex page syntax (`page name: ui:`) but not simple syntax
- Simple pages (`page "Name" at "/route":`) don't support charts

**Example that FAILS**:
```namel3ss
page "Analytics" at "/analytics":
    show card "Performance":
        item:
            sections:
                - type: chart
                  chart_type: line
                  x_axis: date
                  y_axis: score
```

**Impact**: No data visualization in pages
**Workaround**: Use text or external charting
**Fix Needed**: Add chart support to simple page syntax

---

### 5. **Info Grid and Stats Grid Limitations**
**Issue**: Grid sections with complex nested structures fail
- `info_grid` and `stats_grid` types parse but nested items don't work
- Cannot access dataset fields in grid items

**Example that PARTIALLY WORKS**:
```namel3ss
sections:
    - type: info_grid
      columns: 3
      items:
          - icon: "📊"
            label: "Score"
            values:
                - field: average_score  # Field binding fails
```

**Impact**: Cannot create dashboard-style layouts with real data
**Workaround**: Use simple text
**Fix Needed**: Implement field binding in grid items

---

### 6. **Dataset Field Interpolation in Cards**
**Issue**: Template syntax `{{field_name}}` doesn't reliably bind to dataset fields
- Works in some contexts, fails in others
- No validation of field names against dataset schema

**Example**:
```namel3ss
show card "Quiz" from dataset quizzes:
    item:
        header:
            title: "{{title}}"  # May or may not work
            badges:
                - field: difficulty  # Different syntax, also unreliable
```

**Impact**: Cannot display dynamic data from datasets
**Workaround**: Use static text
**Fix Needed**: Implement consistent field binding and validation

---

### 7. **Memory Declarations Require Quotes**
**Issue**: Memory names must be quoted strings, inconsistent with other identifiers

**Example**:
```namel3ss
# FAILS
memory quiz_context:
    type: conversation

# WORKS
memory "quiz_context":
    type: conversation
```

**Impact**: Inconsistent syntax, confusion
**Workaround**: Always use quotes
**Fix Needed**: Allow unquoted identifiers like datasets

---

### 8. **Prompt Definitions Not Supported**
**Issue**: No prompt template syntax in either parser
- Cannot define reusable prompt templates
- LLM prompts must be hardcoded in agent definitions

**Desired Syntax**:
```namel3ss
prompt quiz_generation:
    model: gpt-4
    template: """
    Generate {{num_questions}} quiz questions on {{topic}}.
    Difficulty: {{difficulty}}
    """
```

**Impact**: No prompt management in DSL
**Workaround**: Hardcode prompts or manage externally
**Fix Needed**: Implement prompt declaration parsing

---

### 9. **Form Components Don't Exist**
**Issue**: No `show_form` or form-related components
- Critical for user input (quiz creation, answer submission)
- Referenced in documentation but not implemented

**Desired Syntax**:
```namel3ss
show form "Quiz Builder":
    field "title" type "text" required
    field "difficulty" type "select" options ["easy", "medium", "hard"]
    field "num_questions" type "number"
    submit "Generate Quiz"
```

**Impact**: No user input forms
**Workaround**: External form handling
**Fix Needed**: Implement form component parser and renderer

---

### 10. **Evaluation Result Component Missing**
**Issue**: No component to display grading results
- Needed to show scores, feedback, rubric breakdown
- Critical for grading workflow

**Desired Syntax**:
```namel3ss
show evaluation_result from submission:
    score: "{{points}}/{{max_points}}"
    rubric_breakdown:
        - criterion: "{{name}}"
          score: "{{score}}"
          feedback: "{{comment}}"
```

**Impact**: Cannot display grading results
**Workaround**: Use text or custom components
**Fix Needed**: Implement evaluation_result component

---

### 11. **Chat Thread Component Not Implemented**
**Issue**: No chat/conversation UI component
- Needed for tutoring agent interactions
- Critical for student-AI dialogue

**Desired Syntax**:
```namel3ss
show chat_thread agent explanation_tutor:
    context: "{{question_id}}"
    initial_message: "Need help understanding this question?"
```

**Impact**: No AI chat interfaces
**Workaround**: External chat UI
**Fix Needed**: Implement chat component

---

### 12. **Data Table Component Fails in Simple Pages**
**Issue**: `data_table` not recognized as page statement in simple syntax
- Error: "Unknown page statement: 'data_table'"
- Works in complex page syntax only

**Example that FAILS**:
```namel3ss
page "Library" at "/library":
    data_table from dataset quizzes:
        columns: ["title", "questions", "created"]
```

**Impact**: Cannot display tabular data
**Workaround**: Use cards or external tables
**Fix Needed**: Add data_table to simple page statement parsers

---

### 13. **Tool Call View Component Missing**
**Issue**: No component to display AI tool execution logs
- Useful for debugging and transparency
- Shows which tools agents called

**Impact**: No visibility into AI operations
**Fix Needed**: Implement tool_call_view component

---

### 14. **Agent Panel Component Missing**
**Issue**: No UI component to monitor agent status
- Cannot show agent thinking/reasoning
- No progress indicators for agent tasks

**Impact**: Black box AI execution
**Fix Needed**: Implement agent_panel component

---

### 15. **Diff View Component Missing**
**Issue**: No component to show answer comparisons
- Needed to compare student answer vs correct answer
- Useful for grading workflow

**Desired Syntax**:
```namel3ss
show diff_view:
    left: "{{student_answer}}"
    right: "{{correct_answer}}"
    mode: "side-by-side"
```

**Impact**: Manual answer comparison
**Fix Needed**: Implement diff_view component

---

### 16. **Log View Component Missing**
**Issue**: No component for system/application logs
- Useful for admin/debugging interfaces
- Cannot display execution logs

**Impact**: No log visibility
**Fix Needed**: Implement log_view component

---

## Summary by Category

### Critical (Blocks Core Functionality)
1. ✅ Tool definitions (dual parser conflict)
2. ✅ Agent definitions (dual parser conflict)
3. ✅ Form components (no user input)
4. ✅ Data tables (cannot display tabular data)

### High Priority (Limits User Experience)
5. ✅ Chart components (no visualization)
6. ✅ Evaluation results (grading workflow)
7. ✅ Chat threads (AI tutoring)
8. ✅ Card actions/buttons (no interactivity)

### Medium Priority (Quality of Life)
9. ⚠️ Field interpolation consistency
10. ⚠️ Info/stats grid field binding
11. ⚠️ Prompt templates
12. ⚠️ Diff view

### Low Priority (Nice to Have)
13. 💡 Tool call view
14. 💡 Agent panel
15. 💡 Log view
16. 💡 Memory syntax consistency

---

## Implementation Roadmap

### Phase 1: Parser Unification (Foundation)
**Goal**: Resolve dual parser conflicts
1. Make modern parser primary OR fix legacy parser
2. Standardize tool/agent syntax
3. Document official syntax in grammar spec
4. Update all examples to use consistent syntax

### Phase 2: Core Components (Essential Functionality)
**Goal**: Enable basic user workflows
1. Implement form components (input handling)
2. Fix data_table in simple pages
3. Add chart component support
4. Implement evaluation_result component

### Phase 3: AI Integration (Advanced Features)
**Goal**: Full AI agent support
1. Implement chat_thread component
2. Fix tool/agent declarations to work reliably
3. Add prompt template system
4. Implement agent_panel for monitoring

### Phase 4: Polish (Enhanced UX)
**Goal**: Professional user experience
1. Add card actions/buttons
2. Implement diff_view
3. Fix field interpolation consistency
4. Add tool_call_view and log_view

### Phase 5: Documentation & Testing
**Goal**: Production readiness
1. Update all examples with working complex syntax
2. Comprehensive test coverage for each component
3. Migration guide for existing DSL files
4. Performance optimization

---

## Recommended Next Steps

1. **Immediate**: Choose modern or legacy parser as primary
2. **Week 1-2**: Fix tool/agent syntax (highest ROI)
3. **Week 3-4**: Implement form and data_table components
4. **Month 2**: Add chart and evaluation components
5. **Month 3**: Full AI integration (chat, agents, prompts)

---

## Testing Strategy

For each fix:
1. ✅ Create DSL example demonstrating syntax
2. ✅ Add parser tests
3. ✅ Add IR generation tests
4. ✅ Add codegen tests
5. ✅ Add end-to-end integration test
6. ✅ Update documentation

---

## Success Criteria

Complex Education Quiz Suite example should support:
- ✅ 4 AI tools defined in DSL
- ✅ 3 AI agents with tool access
- ✅ Forms for quiz creation and submission
- ✅ Data tables for quiz/submission lists
- ✅ Charts for analytics visualization
- ✅ Evaluation results display
- ✅ Chat interface for tutoring
- ✅ Interactive card actions
- ✅ All features parse and compile without errors
- ✅ Full pipeline: DSL → Parser → IR → Codegen → Working app
