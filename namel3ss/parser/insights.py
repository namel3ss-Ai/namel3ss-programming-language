from __future__ import annotations

import re
import shlex
from typing import Any, Dict, List, Optional

from namel3ss.ast import (
    Expression,
    Insight,
    InsightAssignment,
    InsightAudience,
    InsightDatasetRef,
    InsightDeliveryChannel,
    InsightEmit,
    InsightLogicStep,
    InsightMetric,
    InsightNarrative,
    InsightSelect,
    InsightThreshold,
    DatasetTransformStep,
)

from .expressions import ExpressionParserMixin
# KeywordRegistry import removed - class does not exist


class InsightParserMixin(ExpressionParserMixin):
    """
    Mixin for parsing business insight declarations.
    
    This parser handles comprehensive insight definitions that combine data analysis,
    metric computation, threshold monitoring, narrative generation, and multi-channel
    delivery. Insights transform raw data into actionable business intelligence.
    
    Syntax Example:
        insight "Sales Performance" from dataset sales:
            logic:
                select sales where region = "North"
                total_revenue = sum(amount)
            
            metrics:
                revenue:
                    value: total_revenue
                    format: currency
                    target: 100000
                growth:
                    value: (current - baseline) / baseline
                    format: percentage
            
            thresholds:
                low_revenue:
                    metric: revenue
                    operator: <
                    value: 50000
                    level: warning
                    message: "Revenue below target"
            
            narratives:
                summary:
                    template: "Total revenue is {revenue} with {growth} growth"
            
            audiences:
                executives:
                    persona: executive
                    channels: dashboard, email
            
            channels:
                dashboard:
                    kind: web
                    target: /sales-dashboard
    
    Features:
        - Data logic with selections, filters, and computations
        - Computed metrics with formatting and baselines
        - Alert thresholds with configurable operators
        - Natural language narrative templates
        - Audience targeting and persona management
        - Multi-channel delivery (dashboard, email, slack, etc.)
        - Dataset references and transformations
        - Parameter configuration
    
    Supported Constructs:
        - logic: Data processing and computation steps
        - compute: Variable assignments and calculations
        - metrics: Key performance indicators
        - thresholds/alerts: Monitoring rules
        - narratives: Text generation templates
        - expose: Public interface for insight data
        - datasets: Additional data source references
        - parameters: Configuration parameters
        - audiences: Target user groups
        - channels: Delivery mechanisms
        - tags: Categorization metadata
    """

    def _parse_insight(self, line: str, line_no: int, base_indent: int) -> Insight:
        """
        Parse an insight declaration.
        
        Insights analyze datasets and generate actionable intelligence with
        metrics, alerts, narratives, and delivery configurations.
        
        Syntax: insight "Name" from dataset DATASET:
        
        Args:
            line: The insight declaration line
            line_no: Current line number
            base_indent: Indentation level of the insight declaration
        
        Returns:
            Insight AST node
        """
        match = re.match(r'insight\s+"([^"]+)"\s+from\s+dataset\s+([^:\s]+)\s*:?', line.strip())
        if not match:
            raise self._error(
                "Expected: insight \"Name\" from dataset DATASET:",
                line_no,
                line,
                hint='Insights require a name and source dataset, e.g., insight "Sales Analysis" from dataset sales:'
            )
        name = match.group(1)
        source_dataset = match.group(2)
        insight = Insight(name=name, source_dataset=source_dataset)

        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            
            # Centralized indentation validation
            self._expect_indent_greater_than(
                base_indent,
                nxt,
                line_no,
                context="insight body",
                hint="Insight directives (logic, metrics, thresholds, etc.) must be indented under the insight declaration"
            )
            lowered = stripped.lower()
            if lowered.startswith('logic:'):
                block_indent = indent
                self._advance()
                steps = self._parse_insight_logic(block_indent)
                insight.logic.extend(steps)
            elif lowered.startswith('compute:'):
                block_indent = indent
                self._advance()
                assignments = self._parse_insight_compute(block_indent)
                if assignments:
                    insight.logic.extend(assignments)
            elif lowered.startswith('metrics:'):
                block_indent = indent
                self._advance()
                metrics = self._parse_insight_metrics(block_indent)
                if metrics:
                    insight.metrics.extend(metrics)
            elif lowered.startswith('thresholds:') or lowered.startswith('alerts:'):
                block_indent = indent
                self._advance()
                thresholds = self._parse_insight_thresholds(block_indent)
                if thresholds:
                    insight.thresholds.extend(thresholds)
                    for threshold in thresholds:
                        insight.alert_thresholds[threshold.name] = {
                            'metric': threshold.metric,
                            'operator': threshold.operator,
                            'level': threshold.level,
                            'message': threshold.message,
                            'window': threshold.window,
                            'extras': dict(threshold.extras),
                        }
            elif lowered.startswith('narratives:'):
                block_indent = indent
                self._advance()
                narratives = self._parse_insight_narratives(block_indent)
                if narratives:
                    insight.narratives.extend(narratives)
            elif lowered.startswith('emit narrative'):
                remainder = stripped[len('emit narrative'):].strip()
                if remainder.startswith(':'):
                    remainder = remainder[1:].strip()
                else:
                    remainder = ''
                self._advance()
                emit_entries: List[InsightEmit] = []
                if remainder:
                    value = self._coerce_scalar(remainder)
                    emit_entries.append(
                        InsightEmit(
                            kind='narrative',
                            content=self._stringify_value(value),
                            props={},
                        )
                    )
                more_emits = self._parse_insight_emit_block(indent, kind='narrative')
                emit_entries.extend(more_emits)
                if emit_entries:
                    insight.logic.extend(emit_entries)
            elif lowered.startswith('expose:'):
                block_indent = indent
                self._advance()
                mapping = self._parse_kv_block(block_indent)
                for key, value in mapping.items():
                    expr = self._coerce_expression(value)
                    insight.expose_as[key] = expr
            elif lowered.startswith('datasets:'):
                block_indent = indent
                self._advance()
                refs = self._parse_insight_datasets(block_indent)
                insight.datasets.extend(refs)
            elif lowered.startswith('parameters:'):
                block_indent = indent
                self._advance()
                params = self._parse_kv_block(block_indent)
                for key, value in params.items():
                    insight.parameters[key] = self._coerce_expression(value)
            elif lowered.startswith('audiences:'):
                block_indent = indent
                self._advance()
                audiences = self._parse_insight_audiences(block_indent)
                insight.audiences.extend(audiences)
            elif lowered.startswith('channels:'):
                block_indent = indent
                self._advance()
                channels = self._parse_insight_channels(block_indent)
                insight.channels.extend(channels)
            elif lowered.startswith('tags:'):
                tag_text = stripped[len('tags:'):].strip()
                insight.tags = self._ensure_string_list(tag_text)
                self._advance()
            elif lowered.startswith('metadata:'):
                block_indent = indent
                self._advance()
                metadata = self._parse_kv_block(block_indent)
                insight.metadata.update(metadata)
            else:
                raise self._error(
                    "Unknown directive inside insight block",
                    self.pos + 1,
                    nxt,
                    hint='Valid insight directives: logic, compute, metrics, thresholds, narratives, emit narrative, expose, datasets, parameters, audiences, channels, tags, metadata'
                )

        return insight

    def _parse_insight_metrics(self, parent_indent: int) -> List[InsightMetric]:
        """
        Parse insight metrics configuration.
        
        Metrics define key performance indicators with values, formatting,
        baselines, targets, and time windows.
        """
        raw_mapping = self._parse_kv_block(parent_indent)
        metrics: List[InsightMetric] = []
        for name, raw in raw_mapping.items():
            config = dict(raw) if isinstance(raw, dict) else {}

            value_source: Any = config.pop('value', None)
            if value_source is None:
                value_source = config.pop('expr', None)
            if value_source is None:
                value_source = config.pop('expression', None)
            if value_source is None:
                for candidate in ('metric', 'measure', 'column'):
                    if candidate in config:
                        value_source = config.pop(candidate)
                        break
            if value_source is None and not isinstance(raw, dict):
                value_source = raw

            label = config.pop('label', config.pop('title', None))
            format_spec = config.pop('format', config.pop('formatter', None))
            unit = config.pop('unit', config.pop('units', None))
            baseline_source = config.pop('baseline', config.pop('compare_to', config.pop('previous', None)))
            target_source = config.pop('target', config.pop('goal', None))
            window_raw = config.pop('window', config.pop('period', None))

            metric = InsightMetric(
                name=name,
                value=self._coerce_expression(value_source),
                label=str(label) if label is not None else None,
                format=str(format_spec) if format_spec is not None else None,
                unit=str(unit) if unit is not None else None,
                baseline=self._coerce_expression(baseline_source) if baseline_source is not None else None,
                target=self._coerce_expression(target_source) if target_source is not None else None,
                window=self._coerce_int(window_raw),
                extras={k: v for k, v in config.items()},
            )
            metrics.append(metric)
        return metrics

    def _parse_insight_thresholds(self, parent_indent: int) -> List[InsightThreshold]:
        """
        Parse insight threshold/alert configurations.
        
        Thresholds monitor metrics and trigger alerts when conditions are met,
        supporting operators like >=, <=, >, <, ==, !=.
        """
        raw_mapping = self._parse_kv_block(parent_indent)
        thresholds: List[InsightThreshold] = []
        for name, raw in raw_mapping.items():
            config = dict(raw) if isinstance(raw, dict) else {}

            metric_name = config.pop('metric', config.pop('for', None))
            operator = config.pop('operator', config.pop('op', None)) or '>='
            level = config.pop('level', config.pop('severity', 'warning'))
            message = config.pop('message', config.pop('text', None))
            window_raw = config.pop('window', config.pop('period', None))
            value_source: Any = config.pop('value', config.pop('threshold', None))

            if value_source is None and not isinstance(raw, dict):
                value_source = raw
            if metric_name is None:
                metric_name = name

            threshold = InsightThreshold(
                name=name,
                metric=str(metric_name),
                operator=str(operator),
                value=self._coerce_expression(value_source) if value_source is not None else None,
                level=str(level) if level is not None else 'warning',
                message=str(message) if message is not None else None,
                window=self._coerce_int(window_raw),
                extras={k: v for k, v in config.items()},
            )
            thresholds.append(threshold)
        return thresholds

    def _parse_insight_datasets(self, parent_indent: int) -> List[InsightDatasetRef]:
        """
        Parse additional dataset references for insights.
        
        Datasets can include transforms and role specifications (source, lookup, etc.).
        """
        mapping = self._parse_kv_block(parent_indent)
        refs: List[InsightDatasetRef] = []
        for key, raw in mapping.items():
            name = key
            role = 'source'
            transforms: List[DatasetTransformStep] = []
            options: Dict[str, Any] = {}
            if isinstance(raw, dict):
                config = dict(raw)
                name_value = config.pop('name', config.pop('dataset', None))
                if name_value is not None:
                    name = str(name_value)
                role_value = config.pop('role', config.pop('type', None))
                if role_value is not None:
                    role = str(role_value)
                transforms_raw = config.pop('transforms', {})
                transforms = self._build_inline_transforms(transforms_raw)
                options = self._coerce_options_dict(config.pop('options', {}))
                if config:
                    options.update(config)
            else:
                if isinstance(raw, str) and raw.strip():
                    name = raw.strip()
            refs.append(
                InsightDatasetRef(
                    name=name,
                    role=role,
                    transforms=transforms,
                    options=options,
                )
            )
        return refs

    def _parse_insight_audiences(self, parent_indent: int) -> List[InsightAudience]:
        """
        Parse audience targeting configuration.
        
        Audiences define user personas with specific needs and preferred channels.
        """
        mapping = self._parse_kv_block(parent_indent)
        audiences: List[InsightAudience] = []
        for name, raw in mapping.items():
            config = dict(raw) if isinstance(raw, dict) else {}
            persona_raw = config.pop('persona', config.pop('profile', None))
            persona = str(persona_raw) if persona_raw is not None else None
            needs_raw = config.pop('needs', {})
            needs = self._coerce_options_dict(needs_raw)
            channels_raw = config.pop('channels', None)
            channels = self._ensure_string_list(channels_raw) if channels_raw is not None else []
            if config:
                extras_bucket = needs.setdefault('extras', {})
                if not isinstance(extras_bucket, dict):
                    extras_bucket = {}
                    needs['extras'] = extras_bucket
                extras_bucket.update(config)
            audiences.append(
                InsightAudience(
                    name=name,
                    persona=persona,
                    needs=needs,
                    channels=channels,
                )
            )
        return audiences

    def _parse_insight_channels(self, parent_indent: int) -> List[InsightDeliveryChannel]:
        """
        Parse delivery channel configurations.
        
        Channels define how insights are delivered (dashboard, email, slack, etc.)
        with scheduling and targeting options.
        """
        mapping = self._parse_kv_block(parent_indent)
        channels: List[InsightDeliveryChannel] = []
        for name, raw in mapping.items():
            if isinstance(raw, dict):
                config = dict(raw)
            else:
                config = {'kind': raw}
            kind_raw = config.pop('kind', config.pop('type', name))
            kind = str(kind_raw) if kind_raw is not None else 'dashboard'
            target_raw = config.pop('target', config.pop('destination', None))
            target = str(target_raw) if target_raw is not None else None
            schedule_raw = config.pop('schedule', config.pop('frequency', None))
            schedule = str(schedule_raw) if schedule_raw is not None else None
            options = self._coerce_options_dict(config.pop('options', {}))
            if config:
                options.update(config)
            options.setdefault('name', name)
            channels.append(
                InsightDeliveryChannel(
                    kind=kind,
                    target=target,
                    schedule=schedule,
                    options=options,
                )
            )
        return channels

    def _build_inline_transforms(self, raw: Any) -> List[DatasetTransformStep]:
        steps: List[DatasetTransformStep] = []
        if isinstance(raw, dict):
            items = raw.items()
        elif isinstance(raw, list):
            items = []
            for index, entry in enumerate(raw):
                if isinstance(entry, dict):
                    entry_map = dict(entry)
                    name = str(entry_map.pop('name', f'step_{index + 1}'))
                    items.append((name, entry_map))
                else:
                    items.append((f'step_{index + 1}', {'expression': entry}))
        else:
            return steps

        for name, entry_raw in items:  # type: ignore[arg-type]
            config = dict(entry_raw) if isinstance(entry_raw, dict) else {'expression': entry_raw}
            transform_type_raw = config.pop('type', config.pop('transform_type', 'custom'))
            transform_type = str(transform_type_raw) if transform_type_raw is not None else 'custom'
            inputs_raw = config.pop('inputs', config.pop('input', []))
            inputs = self._ensure_string_list(inputs_raw)
            output_raw = config.pop('output', None)
            output = str(output_raw) if output_raw is not None else None
            expr_raw = config.pop('expression', config.pop('expr', None))
            expression = self._coerce_expression(expr_raw) if expr_raw is not None else None
            options = self._coerce_options_dict(config.pop('options', {}))
            if config:
                options.update(config)
            steps.append(
                DatasetTransformStep(
                    name=name,
                    transform_type=transform_type,
                    inputs=inputs,
                    output=output,
                    expression=expression,
                    options=options,
                )
            )
        return steps

    def _parse_insight_narratives(self, parent_indent: int) -> List[InsightNarrative]:
        """
        Parse narrative text templates.
        
        Narratives convert metrics into natural language descriptions with
        template interpolation and styling options.
        """
        raw_mapping = self._parse_kv_block(parent_indent)
        narratives: List[InsightNarrative] = []
        for name, raw in raw_mapping.items():
            if isinstance(raw, dict):
                config = dict(raw)
                template_value = config.pop('template', config.pop('text', config.pop('value', '')))
                variant = config.pop('variant', config.pop('type', None))
                style_cfg = config.pop('style', config.pop('styles', {}))
                style = dict(style_cfg) if isinstance(style_cfg, dict) else {}
                extras = {k: v for k, v in config.items()}
            else:
                template_value = raw
                variant = None
                style = {}
                extras = {}

            narrative = InsightNarrative(
                name=name,
                template=str(template_value) if template_value is not None else '',
                variant=str(variant) if variant not in (None, '') else None,
                style=style,
                extras=extras,
            )
            narratives.append(narrative)
        return narratives

    def _parse_insight_compute(self, parent_indent: int) -> List[InsightAssignment]:
        """
        Parse compute block with variable assignments.
        
        Compute blocks define calculations using assignment syntax:
        variable_name = expression
        """
        assignments: List[InsightAssignment] = []
        while self.pos < len(self.lines):
            line = self._peek()
            if line is None:
                break
            indent = self._indent(line)
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= parent_indent:
                break
            if '=' not in stripped:
                raise self._error(
                    "Expected assignment inside compute block",
                    self.pos + 1,
                    line,
                    hint='Use assignment syntax: variable_name = expression'
                )
            eq_index = stripped.find('=')
            if eq_index <= 0:
                raise self._error(
                    "Invalid assignment syntax inside compute block",
                    self.pos + 1,
                    line,
                    hint='Ensure variable name comes before = sign'
                )
            name = stripped[:eq_index].strip()
            expression_text = stripped[eq_index + 1:].strip()
            expression = self._parse_expression(expression_text)
            assignments.append(InsightAssignment(name=name, expression=expression))
            self._advance()
        return assignments

    def _parse_insight_emit_block(self, parent_indent: int, *, kind: str) -> List[InsightEmit]:
        """
        Parse emit block for narrative or data output.
        
        Emit blocks generate output content to be delivered through channels.
        """
        emits: List[InsightEmit] = []
        while self.pos < len(self.lines):
            line = self._peek()
            if line is None:
                break
            indent = self._indent(line)
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= parent_indent:
                break
            value = self._coerce_scalar(stripped)
            emits.append(InsightEmit(kind=kind, content=self._stringify_value(value), props={}))
            self._advance()
        return emits

    def _parse_insight_logic(self, parent_indent: int) -> List[InsightLogicStep]:
        """
        Parse insight logic block with data operations.
        
        Logic blocks support:
        - Assignments: variable = expression
        - Selections: select DATASET where CONDITION order by COLUMNS limit N
        - Emissions: emit KIND content
        """
        steps: List[InsightLogicStep] = []
        while self.pos < len(self.lines):
            line = self._peek()
            if line is None:
                break
            indent = self._indent(line)
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= parent_indent:
                break

            if '=' in stripped:
                eq_index = stripped.find('=')
                before = stripped[: max(eq_index, 0)]
                after = stripped[eq_index + 1 :]
                if eq_index > 0 and (
                    eq_index == len(stripped) - 1
                    or stripped[eq_index + 1] != '='
                ) and stripped[eq_index - 1] not in {'!', '<', '>'}:
                    name = before.strip()
                    expression_text = after.strip()
                    expression = self._parse_expression(expression_text)
                    steps.append(InsightAssignment(name=name, expression=expression))
                    self._advance()
                    continue

            if stripped.lower().startswith('select '):
                select_body = stripped[7:]
                lower_body = select_body.lower()

                order_idx = lower_body.find(' order by ')
                order_by: Optional[List[str]] = None
                if order_idx != -1:
                    order_by_text = select_body[order_idx + len(' order by ') :]
                    order_by = [c.strip() for c in order_by_text.split(',') if c.strip()]
                    select_body = select_body[:order_idx]
                    lower_body = select_body.lower()

                limit_idx = lower_body.find(' limit ')
                limit: Optional[int] = None
                if limit_idx != -1:
                    limit_text = select_body[limit_idx + len(' limit ') :].strip()
                    try:
                        limit = int(limit_text.split()[0])
                    except (ValueError, IndexError):
                        limit = None
                    select_body = select_body[:limit_idx]
                    lower_body = select_body.lower()

                where_idx = lower_body.find(' where ')
                condition: Optional[Expression] = None
                if where_idx != -1:
                    condition_text = select_body[where_idx + len(' where ') :].strip()
                    condition = self._parse_expression(condition_text)
                    select_body = select_body[:where_idx]

                dataset_name = select_body.strip()
                steps.append(
                    InsightSelect(
                        dataset=dataset_name,
                        condition=condition,
                        limit=limit,
                        order_by=order_by,
                    )
                )
                self._advance()
                continue

            if stripped.lower().startswith('emit '):
                emit_body = stripped[5:]
                parts = shlex.split(emit_body)
                if not parts:
                    raise self._error(
                        "Emit statements must specify a kind",
                        self.pos + 1,
                        line,
                        hint='Use: emit narrative "text" or emit data variable_name'
                    )
                kind = parts[0]
                content = ' '.join(parts[1:]) if len(parts) > 1 else ''
                steps.append(InsightEmit(kind=kind, content=content))
                self._advance()
                continue

            raise self._error(
                "Unknown insight logic directive",
                self.pos + 1,
                line,
                hint='Valid logic operations: assignments (var = expr), select statements, emit statements'
            )

        return steps
