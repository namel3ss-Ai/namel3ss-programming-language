# RFC XXXX: [Title]

- **Status**: Draft
- **Author**: [Your Name] (@username)
- **Created**: YYYY-MM-DD
- **Updated**: YYYY-MM-DD
- **Tracking Issue**: #NNN (create after RFC is filed)

## Summary

> One-paragraph explanation of the feature/change.

## Motivation

> Why are we doing this? What use cases does it support? What problems does it solve?
>
> Include concrete examples if applicable.

## Detailed Design

> This is the bulk of the RFC. Explain the design in enough detail that:
> - Someone familiar with Namel3ss can understand the proposal
> - Someone implementing it can do so from this spec
> - Corner cases are addressed

### Syntax

If this RFC involves new syntax, provide:
- Grammar rules (in EBNF or similar)
- Example code snippets
- Explanation of tokens and parsing

```namel3ss
// Example syntax
app "Example" {
    // feature demonstration
}
```

### Semantics

Explain the behavior and meaning of the proposed feature:
- What does it do at runtime?
- How does it interact with other features?
- What are the edge cases?

### Type System Impact

If this affects the type system:
- How are types inferred/checked?
- What type errors can occur?
- Examples of type-correct and type-incorrect usage

### Conformance Tests

List the conformance tests that will be added (brief descriptions):

1. `test-id-001`: Parse valid usage
2. `test-id-002`: Reject invalid syntax
3. `test-id-003`: Runtime behavior verification

(Actual test descriptors will be created during implementation)

## Alternatives Considered

> What other designs were considered? Why was this approach chosen?
>
> Include trade-offs and comparisons.

## Backwards Compatibility

> Is this a breaking change? If so:
> - What existing code will break?
> - How can users migrate?
> - Can we provide deprecation warnings?
>
> If not breaking, explain how backwards compatibility is maintained.

## Implementation Plan

> Rough outline of implementation steps:
> 1. Update parser to recognize new syntax
> 2. Add AST nodes
> 3. Implement semantic analysis
> 4. Add conformance tests
> 5. Update documentation

### Estimated Effort

> Small / Medium / Large
>
> Brief justification of complexity

## Future Extensions

> How might this feature be extended in the future?
> What related features might build on this?

## Unresolved Questions

> What parts of the design are still to be determined?
> What do we need to figure out during implementation?

## References

> Links to related issues, discussions, prior art, research papers, etc.

---

## Discussion Log

### YYYY-MM-DD: Initial discussion
- [Summary of discussion points]

### YYYY-MM-DD: Revision based on feedback
- [What changed and why]

---

## Decision

**Status**: [Draft / Accepted / Rejected / Implemented]

**Decided**: YYYY-MM-DD

**Rationale**: [Why this was accepted/rejected]

**Implementation Tracking**: #NNN (issue tracking implementation)
