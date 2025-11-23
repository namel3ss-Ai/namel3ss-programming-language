"""Static analysis utilities for effect inference and enforcement."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set

from namel3ss.ast import (
    Action,
    ActionOperationType,
    App,
    AskConnectorOperation,
    Chain,
    ChainStep,
    ForLoop,
    IfBlock,
    LogStatement,
    Page,
    RunChainOperation,
    RunPromptOperation,
    ShowForm,
    Statement,
    WhileLoop,
    WorkflowForBlock,
    WorkflowIfBlock,
    WorkflowNode,
    WorkflowWhileBlock,
)
from namel3ss.errors import N3EffectError

AI_EFFECT = "ai"
ALLOWED_DECLARED_EFFECTS: Set[str] = {"pure", AI_EFFECT}


class EffectError(N3EffectError):
    """Raised when an effect constraint is violated."""


class EffectAnalyzer:
    """Infer and validate effect annotations for the application AST."""

    def __init__(self, app: App):
        self.app = app
        self.chain_effects: Dict[str, Set[str]] = {}

    def analyze(self) -> None:
        """Infer effects for known constructs and enforce declared restrictions."""

        self._mark_prompts()
        self._analyze_chains()
        self._analyze_pages()

    # ------------------------------------------------------------------
    # Prompt handling
    # ------------------------------------------------------------------
    def _mark_prompts(self) -> None:
        for prompt in self.app.prompts:
            prompt.effects = {AI_EFFECT}

    # ------------------------------------------------------------------
    # Chain analysis
    # ------------------------------------------------------------------
    def _analyze_chains(self) -> None:
        for chain in self.app.chains:
            effects = self._infer_chain_effects(chain)
            chain.effects = effects
            self.chain_effects[chain.name] = effects
            self._validate_declared_effect(
                declared=chain.declared_effect,
                actual=effects,
                context=f"Chain '{chain.name}'",
            )

    def _infer_chain_effects(self, chain: Chain) -> Set[str]:
        return self._effects_from_workflow_nodes(chain.steps)

    def _effects_from_workflow_nodes(self, nodes: Sequence[WorkflowNode]) -> Set[str]:
        effects: Set[str] = set()
        for node in nodes or []:
            effects.update(self._effects_for_node(node))
        return effects

    def _effects_for_node(self, node: WorkflowNode) -> Set[str]:
        if isinstance(node, ChainStep):
            kind = (node.kind or "").lower()
            if kind in {"connector", "prompt", "llm"}:
                return {AI_EFFECT}
            return set()
        if isinstance(node, WorkflowIfBlock):
            effects = self._effects_from_workflow_nodes(node.then_steps)
            for _, branch_steps in node.elif_steps:
                effects.update(self._effects_from_workflow_nodes(branch_steps))
            effects.update(self._effects_from_workflow_nodes(node.else_steps))
            return effects
        if isinstance(node, WorkflowForBlock):
            return self._effects_from_workflow_nodes(node.body)
        if isinstance(node, WorkflowWhileBlock):
            return self._effects_from_workflow_nodes(node.body)
        return set()

    # ------------------------------------------------------------------
    # Page traversal and action/form analysis
    # ------------------------------------------------------------------
    def _analyze_pages(self) -> None:
        for page in self.app.pages:
            self._visit_statements(page, page.statements)

    def _visit_statements(self, page: Page, statements: Sequence[Statement]) -> None:
        for statement in statements:
            if isinstance(statement, Action):
                self._analyze_action(page, statement)
            elif isinstance(statement, ShowForm):
                self._analyze_form(page, statement)
            elif isinstance(statement, LogStatement):
                # Log statements are side effects but are generally benign
                # They don't require special effect validation
                pass  
            elif isinstance(statement, IfBlock):
                self._visit_statements(page, statement.body)
                for branch in statement.elifs:
                    self._visit_statements(page, branch.body)
                if statement.else_body:
                    self._visit_statements(page, statement.else_body)
            elif isinstance(statement, ForLoop):
                self._visit_statements(page, statement.body)
            elif isinstance(statement, WhileLoop):
                self._visit_statements(page, statement.body)

    def _analyze_action(self, page: Page, action: Action) -> None:
        context = f"Action '{action.name}' on page '{page.name}'"
        effects = self._effects_for_operations(action.operations, context)
        action.effects = effects
        self._validate_declared_effect(
            declared=action.declared_effect,
            actual=effects,
            context=context,
        )

    def _analyze_form(self, page: Page, form: ShowForm) -> None:
        context = f"Form '{form.title}' on page '{page.name}'"
        form.effects = self._effects_for_operations(form.on_submit_ops, context)

    # ------------------------------------------------------------------
    # Effect utilities
    # ------------------------------------------------------------------
    def _effects_for_operations(
        self,
        operations: Iterable[ActionOperationType],
        context: str,
    ) -> Set[str]:
        effects: Set[str] = set()
        for operation in operations:
            if isinstance(operation, AskConnectorOperation):
                effects.add(AI_EFFECT)
            elif isinstance(operation, RunPromptOperation):
                effects.add(AI_EFFECT)
            elif isinstance(operation, RunChainOperation):
                chain_name = operation.chain_name
                chain_effect = self.chain_effects.get(chain_name)
                if chain_effect is None:
                    raise EffectError(
                        f"{context} references undefined chain '{chain_name}'."
                    )
                effects.update(chain_effect)
        return effects

    def _validate_declared_effect(
        self,
        *,
        declared: Optional[str],
        actual: Set[str],
        context: str,
    ) -> None:
        if declared is None:
            return
        if declared not in ALLOWED_DECLARED_EFFECTS:
            allowed = ", ".join(sorted(ALLOWED_DECLARED_EFFECTS))
            raise EffectError(f"{context} declares unsupported effect '{declared}'. Allowed: {allowed}.")
        if declared == "pure" and AI_EFFECT in actual:
            raise EffectError(f"{context} is declared pure but performs AI operations.")
