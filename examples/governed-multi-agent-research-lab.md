# Governed Multi-Agent Research Lab

## Overview

The **Governed Multi-Agent Research Lab** is a production-grade example demonstrating enterprise AI governance, safety monitoring, and policy enforcement in a multi-agent collaborative system. This application showcases how to build AI systems with centralized oversight, comprehensive audit trails, and real-time safety checks.

### Key Features

- **🎯 Central Governance**: Dedicated governance agent supervises all other agents
- **👥 Multi-Agent Collaboration**: 6 specialized agents working together on research tasks
- **🛡️ Policy Enforcement**: Real-time policy compliance checking and risk assessment
- **📊 Comprehensive Observability**: Full audit trails, performance metrics, and violation logging
- **🔍 Safety Monitoring**: Hallucination detection, bias analysis, and quality scoring
- **🔀 Workflow Orchestration**: Graph-based multi-agent workflows with conditional routing

---

## Architecture

### Agent Hierarchy

```
                    ┌──────────────────────┐
                    │  Governance Agent    │
                    │  (Central Supervisor)│
                    └──────────┬───────────┘
                               │ supervises
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼────────┐  ┌───▼────────┐  ┌───▼──────────┐
    │ Researcher Agent │  │ Critic     │  │ Safety       │
    │                  │  │ Agent      │  │ Reviewer     │
    └─────────┬────────┘  └───┬────────┘  └───┬──────────┘
              │               │                │
    ┌─────────▼────────┐  ┌───▼────────┐      │
    │ Explainer Agent  │  │ Retriever  │      │
    │                  │  │ Agent      │      │
    └──────────────────┘  └────────────┘      │
                                               │
                          All outputs → Safety Check → Governance Approval
```

---

## Component Specifications

### 1. LLM Configurations

The system uses multiple LLM configurations optimized for different roles:

| LLM ID | Model | Temperature | Max Tokens | Purpose |
|--------|-------|-------------|------------|---------|
| **gpt4** | gpt-4 | 0.7 | 3000 | General research and reasoning |
| **gpt4_precise** | gpt-4 | 0.2 | 2000 | Precise analysis and governance |
| **claude** | claude-3-opus | 0.6 | 2500 | Alternative perspective and critique |
| **gpt35_fast** | gpt-3.5-turbo | 0.5 | 2000 | Fast retrieval and initial analysis |

### 2. Specialist Agents

#### Researcher Agent
- **LLM**: gpt4 (temperature 0.7)
- **Role**: Investigate questions, propose evidence-based arguments
- **Tools**: `detect_hallucination`, `score_argument_quality`
- **Memory**: conversation_history (20 items)
- **Max Turns**: 15

#### Critic Agent
- **LLM**: claude (temperature 0.5)
- **Role**: Examine arguments for logical flaws, suggest improvements
- **Tools**: `score_argument_quality`, `calculate_bias_metrics`
- **Memory**: conversation_history
- **Max Turns**: 10

#### Explainer Agent
- **LLM**: gpt4 (temperature 0.7)
- **Role**: Translate complex research into accessible explanations
- **Tools**: `score_argument_quality`
- **Memory**: conversation_history
- **Max Turns**: 10

#### Retriever Agent
- **LLM**: gpt35_fast (temperature 0.4)
- **Role**: Find and synthesize information from sources
- **Tools**: None (focused on retrieval)
- **Memory**: research_context (50 items)
- **Max Turns**: 8

#### Safety Reviewer Agent
- **LLM**: gpt4_precise (temperature 0.3)
- **Role**: Evaluate outputs for potential harms and risks
- **Tools**: `check_policy_compliance`, `detect_hallucination`, `calculate_bias_metrics`
- **Memory**: policy_violations_log (500 items)
- **Max Turns**: 5

#### Governance Agent (Central Supervisor)
- **LLM**: gpt4_precise (temperature 0.2)
- **Role**: Supervise all agents, enforce policies, assess overall quality
- **Tools**: All 5 governance tools
- **Memory**: governance_decisions (1000 items)
- **Max Turns**: 20
- **Authority**: Can approve, request revisions, or block outputs

### 3. Governance Tools

#### check_policy_compliance
Evaluates content against organizational policies and safety guidelines.

**Parameters**:
- `text` (string, required): Content to evaluate
- `context` (string, optional): Additional context about purpose
- `policy_set` (string, optional): Which policy set (general, research, enterprise, academic)

**Returns**:
- `compliant` (boolean): Whether content passes policy checks
- `risk_level` (string): SAFE | LOW_RISK | MEDIUM_RISK | HIGH_RISK | UNSAFE
- `policy_tags` (array): Specific policies triggered
- `violations` (array): List of policy violations found
- `recommendations` (array): Suggestions for achieving compliance
- `confidence_score` (number): Confidence in assessment (0-100)

#### score_argument_quality
Assesses logical reasoning, evidence quality, and argument structure.

**Parameters**:
- `argument` (string, required): The argument text to analyze
- `criteria` (array, optional): Specific criteria to evaluate

**Returns**:
- `overall_score` (number): Overall quality score (0-100)
- `logic_score` (number): Logical coherence rating
- `evidence_score` (number): Evidence quality and sufficiency
- `clarity_score` (number): How clear and well-structured
- `strengths` (array): Identified strong points
- `weaknesses` (array): Areas needing improvement
- `reasoning_gaps` (array): Missing logical steps
- `recommendations` (array): Improvement suggestions

#### compare_model_outputs
Compares outputs from different models to identify consensus and differences.

**Parameters**:
- `output_a` (string, required): First output to compare
- `output_b` (string, required): Second output to compare
- `comparison_criteria` (array, optional): Criteria for comparison

**Returns**:
- `similarity_score` (number): Content similarity (0-100)
- `quality_comparison` (object): Comparative quality scores
- `key_differences` (array): Significant differences identified
- `consensus_points` (array): Points of agreement
- `recommendation` (string): Which output is preferred and why

#### detect_hallucination
Analyzes text for potential hallucinations or unsupported claims.

**Parameters**:
- `text` (string, required): Text to analyze
- `context` (string, optional): Known facts or source material

**Returns**:
- `hallucination_risk` (string): LOW | MEDIUM | HIGH
- `suspicious_claims` (array): Claims that may be unsupported
- `confidence_scores` (object): Confidence in various statements
- `verification_needed` (array): Claims requiring external verification

#### calculate_bias_metrics
Evaluates text for various types of bias.

**Parameters**:
- `text` (string, required): Text to evaluate
- `bias_types` (array, optional): Types to check (demographic, political, confirmation, anchoring)

**Returns**:
- `overall_bias_score` (number): Overall bias level (0-100, lower is better)
- `bias_indicators` (array): Specific bias indicators found
- `neutrality_score` (number): Neutrality rating (0-100)
- `recommendations` (array): Suggestions for reducing bias

### 4. Multi-Agent Workflows

#### Research Debate Workflow
A comprehensive multi-agent research and debate process with governance oversight.

**Graph**: research_debate_workflow  
**Max Hops**: 25  
**Edges**: 7 conditional transitions

**Flow**:
1. **researcher_agent** → **critic_agent** (if research_complete)
2. **critic_agent** → **explainer_agent** (if critique_done)
3. **explainer_agent** → **retriever_agent** (if needs_more_context)
4. **explainer_agent** → **safety_reviewer_agent** (if explanation_ready)
5. **retriever_agent** → **researcher_agent** (if context_retrieved)
6. **safety_reviewer_agent** → **governance_agent** (if safety_check_complete)
7. **governance_agent** → **researcher_agent** (if revision_needed)

**Use Cases**:
- Collaborative research with multiple perspectives
- Iterative argument refinement through critique
- Knowledge synthesis with safety checks
- Governed decision-making with audit trails

#### Quick Review Workflow
A streamlined review process for faster turnaround.

**Graph**: quick_review_workflow  
**Max Hops**: 10  
**Edges**: 4 conditional transitions

**Flow**:
1. **researcher_agent** → **critic_agent** (if quick_analysis_done)
2. **critic_agent** → **safety_reviewer_agent** (if critical_review_complete)
3. **safety_reviewer_agent** → **governance_agent** (if safety_approved)
4. **governance_agent** → **researcher_agent** (if quick_revision_needed)

**Use Cases**:
- Rapid content review and approval
- Fast policy compliance checking
- Quick safety assessments
- Expedited governance decisions

### 5. Memory Systems

| Memory | Type | Capacity | Purpose |
|--------|------|----------|---------|
| **debate_session_history** | short_term | 100 | Track debate session interactions |
| **governance_decisions** | long_term | 1000 | Log all governance agent decisions |
| **policy_violations_log** | long_term | 500 | Record policy violations and resolutions |
| **agent_performance_metrics** | long_term | 2000 | Monitor agent performance over time |
| **research_context** | short_term | 50 | Store relevant research context |
| **conversation_history** | short_term | 20 | Maintain conversation continuity |

### 6. Datasets

| Dataset | Source | Purpose |
|---------|--------|---------|
| **debate_sessions** | table debate_sessions | Store complete debate session records |
| **governance_events** | table governance_events | Track all governance actions and decisions |
| **policy_violations** | table policy_violations | Record violations for analysis and training |
| **active_policies** | table active_policies | Maintain current policy configurations |

---

## Usage Examples

### Example 1: Research Debate Session

```python
# Initialize a research debate on a complex topic
session = start_workflow("research_debate_workflow", {
    "topic": "Impact of AI governance on innovation",
    "depth": "comprehensive",
    "safety_level": "high"
})

# The workflow automatically:
# 1. Researcher investigates and proposes arguments
# 2. Critic examines for logical flaws
# 3. Explainer translates findings
# 4. Retriever fetches supporting context if needed
# 5. Safety reviewer checks for risks
# 6. Governance agent makes final approval
```

### Example 2: Policy Compliance Check

```python
# Check content against policies before publication
result = governance_agent.check_policy("content_to_publish", 
    policy_set="enterprise",
    context="public blog post")

if result.compliant:
    publish(content)
else:
    handle_violations(result.violations)
    revise_content(result.recommendations)
```

### Example 3: Bias Analysis

```python
# Analyze generated content for bias
bias_report = governance_agent.calculate_bias("generated_content",
    bias_types=["demographic", "political", "confirmation"])

if bias_report.overall_bias_score > 30:
    apply_debiasing(bias_report.recommendations)
```

---

## Enterprise AI Safety Features

### 1. Multi-Layer Safety Checks

- **Agent-Level**: Each agent has built-in safety constraints
- **Tool-Level**: Governance tools provide specialized safety analysis
- **Workflow-Level**: Mandatory safety reviewer in critical paths
- **Governance-Level**: Final approval by central governance agent

### 2. Comprehensive Audit Trails

All actions are logged to datasets:
- Agent interactions → debate_sessions
- Governance decisions → governance_events
- Policy violations → policy_violations
- Performance metrics → agent_performance_metrics

### 3. Real-Time Policy Enforcement

- Content scanned before and after agent processing
- Automatic risk level assessment
- Policy violation blocking with clear explanations
- Revision requests with specific recommendations

### 4. Quality Assurance

- Argument quality scoring for logical rigor
- Hallucination detection for factual accuracy
- Model output comparison for consistency
- Bias analysis for fairness

---

## Configuration Guidelines

### Adjusting Safety Levels

**High Safety** (Enterprise/Production):
```
- governance_agent temperature: 0.2
- safety_reviewer_agent temperature: 0.3
- max_turns: Limited to prevent runaway
- policy_set: "enterprise"
```

**Medium Safety** (Research/Development):
```
- governance_agent temperature: 0.3
- safety_reviewer_agent temperature: 0.4
- max_turns: Moderate limits
- policy_set: "research"
```

**Low Safety** (Experimental):
```
- governance_agent temperature: 0.4
- safety_reviewer_agent temperature: 0.5
- max_turns: Higher limits
- policy_set: "general"
```

### Scaling Considerations

**For High-Volume Deployments**:
- Increase memory capacities (governance_decisions: 10000+)
- Use faster LLMs for preliminary checks (gpt35_fast)
- Implement caching for repeated policy checks
- Parallelize independent agent tasks

**For High-Accuracy Requirements**:
- Use gpt4_precise for all critical agents
- Lower temperatures (0.1-0.3)
- Increase max_tokens for detailed reasoning
- Add redundant safety checks

---

## UI Specifications (Future Implementation)

> **Note**: Page declarations are currently not included in the `.ai` file due to parser limitations. These specifications document the intended UI design for future implementation.

### Page 1: Governed Lab Dashboard

**Route**: `/`  
**Purpose**: High-level overview of sessions and governance metrics

**Components**:
- **Session Status Panel**: Active debates, completed sessions, pending reviews
- **Governance Metrics**: Policy compliance rate, violation frequency, approval latency
- **Agent Performance**: Success rates, average turns, quality scores
- **Safety Alerts**: Recent high-risk detections, policy violations, bias warnings
- **Quick Actions**: Start new debate, review pending decisions, view audit logs

### Page 2: Session Detail View

**Route**: `/session/:id`  
**Purpose**: In-depth view of multi-agent debate session

**Components**:
- **Agent Interaction Timeline**: Chronological view of agent exchanges
- **Governance Checkpoints**: Policy checks, safety reviews, approval stages
- **Argument Quality Metrics**: Scores, strengths/weaknesses, recommendations
- **Safety Analysis**: Hallucination risks, bias indicators, compliance status
- **Session Controls**: Pause, resume, intervene, export transcript

### Page 3: Policy Configuration

**Route**: `/policy`  
**Purpose**: Manage governance rules and safety thresholds

**Components**:
- **Active Policies List**: Current policies with enable/disable toggles
- **Policy Editor**: Create/modify policy rules with validation
- **Threshold Configuration**: Set risk levels, bias limits, quality minimums
- **Violation History**: Patterns, frequency, agent associations
- **Policy Testing**: Simulate policy checks on sample content

---

## Deployment Checklist

- [ ] Configure LLM API keys (OpenAI, Anthropic)
- [ ] Set up database tables for datasets
- [ ] Define organization-specific policies
- [ ] Set safety thresholds for risk levels
- [ ] Initialize memory systems with appropriate capacities
- [ ] Test governance tools with sample content
- [ ] Run end-to-end workflow tests
- [ ] Set up monitoring and alerting
- [ ] Configure audit log retention policies
- [ ] Train team on governance intervention procedures

---

## Limitations & Future Enhancements

### Current Limitations

1. **No Real-Time UI**: Page declarations not yet supported in parser
2. **Static Policies**: Policies defined at deployment time
3. **Linear Workflows**: Limited conditional branching in graphs
4. **Memory Constraints**: Fixed capacity limits on memory systems

### Planned Enhancements

1. **Dynamic Policy Learning**: Policies that adapt based on violation patterns
2. **Advanced Graph Features**: Parallel execution, dynamic routing, sub-workflows
3. **Real-Time Dashboards**: Live monitoring of agent interactions
4. **ML-Based Safety**: Train custom models on organization-specific safety criteria
5. **Federation Support**: Multi-organization governance with shared policies
6. **API Gateway**: RESTful API for external integrations

---

## Support & Resources

- **Documentation**: See `NAMEL3SS_DOCUMENTATION.md` for language reference
- **Examples**: Browse `examples/` directory for more patterns
- **Issues**: Report parser or feature issues on GitHub
- **Community**: Join discussions about governed AI systems

---

## License

This example is part of the Namel3ss programming language project. See LICENSE file for details.

---

**Last Updated**: November 2025  
**Version**: 1.0  
**Status**: Production-Ready (Core Features), UI Pending Parser Support
