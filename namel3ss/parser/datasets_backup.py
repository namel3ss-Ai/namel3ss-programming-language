"""Backward compatibility wrapper for dataset parser."""

from __future__ import annotations

# Import the refactored implementation
from .datasets import DatasetParserMixin

# Re-export for backward compatibility
__all__ = ['DatasetParserMixin']
    """
    Mixin for parsing dataset declarations with comprehensive data processing capabilities.
    
    This parser handles complete dataset definitions including source connections,
    transformations, schema specifications, feature engineering, quality checks,
    and operational configurations like caching, refresh policies, and streaming.
    
    Syntax Example:
        dataset "sales_data" from table sales:
            filter by: revenue > 0
            add column profit = revenue - cost
            group by: region, product_category
            order by: date desc
            
            schema:
                region: string
                    description: "Sales region"
                    nullable: false
                revenue: float
                    constraints:
                        min: 0
            
            feature "revenue_feature":
                role: numeric
                source: revenue
                expression: log(revenue + 1)
            
            target "high_value":
                kind: classification
                expression: revenue > 10000
                positive_class: "high"
            
            quality "revenue_check":
                condition: revenue >= 0
                severity: error
                message: "Revenue cannot be negative"
            
            profile:
                row_count: 10000
                column_count: 15
                freshness: "1 hour"
            
            auto refresh every 5 minutes
            tags: sales, analytics, production
    
    Data Sources:
        - table: Existing database tables
        - file: CSV, JSON, Parquet files
        - dataset: Reference other N3 datasets
        - sql: SQL connectors with table/view/query access
        - rest: REST API endpoints with configurable requests
        - Custom connector types
    
    Operations:
        - Filter: filter by CONDITION
        - Compute: add column NAME = EXPRESSION
        - Group: group by COLUMNS
        - Order: order by COLUMNS
        - Aggregate: sum/count/avg/min/max: EXPRESSION
        - Join: join [type] source NAME on CONDITION
        - Window: add column NAME = FUNCTION over FRAME
    
    Configuration:
        - schema: Column definitions with types and constraints
        - transform: Custom transformation steps
        - feature: ML feature engineering specifications
        - target: ML prediction targets
        - quality: Data quality validation rules
        - profile: Dataset statistics and metadata
        - metadata: Custom metadata key-value pairs
        - lineage: Data lineage tracking
        - cache: Caching policies
        - pagination: Pagination configuration
        - stream: Streaming data policies
        - reactive: Real-time reactivity
        - auto refresh: Automatic refresh intervals
        - tags: Categorization tags
    
    Advanced Features:
        - Window functions with frames (ROWS/RANGE BETWEEN)
        - Multi-source joins with type specifications
        - Complex transformations with expressions
        - Quality checks with thresholds and alerts
        - Feature roles (numeric, categorical, text, etc.)
        - Target types (classification, regression, ranking)
        - Connector options with nested configuration
        - Refresh policies (polling, webhook, event-driven)
    """

    def _parse_dataset(self, line: str, line_no: int, base_indent: int) -> Dataset:
        """
        Parse a dataset declaration.
        
        Datasets define data sources with transformations, schema, features,
        quality checks, and operational configurations.
        
        Syntax: dataset "Name" from SOURCE_TYPE SOURCE_REF:
        
        Args:
            line: The dataset declaration line
            line_no: Current line number
            base_indent: Indentation level of the dataset declaration
        
        Returns:
            Dataset AST node
        """
        raw = line.strip()
        if raw.endswith(':'):
            raw = raw[:-1]
        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            raise self._error(
                f"Unable to parse dataset declaration: {exc}",
                line_no,
                line,
                hint='Check for unmatched quotes in dataset declaration'
            )

        if len(parts) < 2 or parts[0] != 'dataset':
            raise self._error(
                "Expected: dataset \"Name\" from ...",
                line_no,
                line,
                hint='Datasets require a name and source, e.g., dataset "Sales" from table sales'
            )
        name = parts[1]
        if len(parts) < 4 or parts[2].lower() != 'from':
            raise self._error(
                "Dataset must specify source via 'from'",
                line_no,
                line,
                hint='Use: dataset "Name" from table/file/sql/rest/dataset SOURCE'
            )

        source_type = parts[3].lower()
        idx = 4
        source = ''
        connector: Optional[DatasetConnectorConfig] = None

        def require_token(token_name: str = 'value') -> str:
            nonlocal idx
            if idx >= len(parts):
                raise self._error(f"Expected {token_name} in dataset declaration", line_no, line)
            value = parts[idx]
            idx += 1
            return value

        if source_type == 'table':
            source = require_token('table name')
            connector = DatasetConnectorConfig(connector_type='table', connector_name=None)
        elif source_type == 'file':
            source = require_token('file path')
            connector = DatasetConnectorConfig(connector_type='file', connector_name=source)
        elif source_type == 'dataset':
            source = require_token('dataset name')
            connector = DatasetConnectorConfig(connector_type='dataset', connector_name=source)
        elif source_type == 'sql':
            connector_name = require_token('SQL connector name')
            options: Dict[str, Any] = {}
            if idx < len(parts):
                mode = parts[idx].lower()
                if mode in {'table', 'view', 'query'}:
                    idx += 1
                    target = require_token(f'{mode} reference')
                    options[mode] = target
                    source = target
            connector = DatasetConnectorConfig(
                connector_type='sql',
                connector_name=connector_name,
                options=options,
            )
            if not source:
                source = connector_name
        elif source_type == 'rest':
            connector_name = require_token('REST connector name')
            options: Dict[str, Any] = {}
            if idx < len(parts) and parts[idx].lower() == 'endpoint':
                idx += 1
                endpoint = require_token('endpoint path')
                options['endpoint'] = endpoint
                source = endpoint
            if idx < len(parts) and parts[idx].lower() == 'method':
                idx += 1
                method = require_token('HTTP method')
                options['method'] = method
            connector = DatasetConnectorConfig(
                connector_type='rest',
                connector_name=connector_name,
                options=options,
            )
            if not source:
                source = connector_name
        else:
            source = require_token('source reference') if idx < len(parts) else name
            connector = DatasetConnectorConfig(connector_type=source_type, connector_name=source)

        dataset = Dataset(name=name, source_type=source_type, source=source, connector=connector)

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
                context="dataset body",
                hint="Dataset operations and configurations must be indented under the dataset declaration"
            )

            lowered = stripped.lower()
            if lowered.startswith('filter by:'):
                cond_text = stripped[len('filter by:'):].strip()
                condition = self._parse_expression(cond_text)
                dataset.operations.append(FilterOp(condition=condition))
                self._advance()
            elif lowered.startswith('add column '):
                col_def = stripped[len('add column '):].strip()
                if '=' not in col_def:
                    raise self._error(
                        "Expected: add column name = expression",
                        self.pos + 1,
                        nxt,
                        hint='Computed columns require an assignment, e.g., add column profit = revenue - cost'
                    )
                col_name, col_expr = col_def.split('=', 1)
                col_name = col_name.strip()
                col_expr = col_expr.strip()
                if not col_name:
                    raise self._error(
                        "Column name cannot be empty",
                        self.pos + 1,
                        nxt,
                        hint='Provide a column name before the = sign'
                    )

                window_match = re.search(r'\bover\b', col_expr)
                if window_match:
                    func_match = re.match(r'([A-Za-z_][\w]*)\(([^)]*)\)\s+over\s+(.+)', col_expr)
                    if func_match:
                        function = func_match.group(1)
                        target = func_match.group(2).strip() or None
                        frame_spec = func_match.group(3).strip()
                        frame = self._parse_window_frame(frame_spec)
                        dataset.operations.append(
                            WindowOp(
                                name=col_name,
                                function=function,
                                target=target,
                                partition_by=None,
                                order_by=None,
                                frame=frame,
                            )
                        )
                        self._advance()
                        continue
                expression = self._parse_expression(col_expr)
                dataset.operations.append(
                    ComputedColumnOp(name=col_name, expression=expression)
                )
                self._advance()
            elif lowered.startswith('group by:'):
                cols = stripped[len('group by:'):].strip()
                col_list = [c.strip() for c in cols.split(',') if c.strip()]
                dataset.operations.append(GroupByOp(columns=col_list))
                self._advance()
            elif lowered.startswith('order by:'):
                cols = stripped[len('order by:'):].strip()
                order_cols = [c.strip() for c in cols.split(',') if c.strip()]
                dataset.operations.append(OrderByOp(columns=order_cols))
                self._advance()
            elif re.match(r'(sum|count|avg|min|max):', lowered):
                func, expr = stripped.split(':', 1)
                expr = expr.strip()
                dataset.operations.append(
                    AggregateOp(function=func.strip(), expression=expr)
                )
                self._advance()
            elif lowered.startswith('join '):
                join_match = re.match(
                    r'join\s+(?:(inner|left|right|full)\s+)?(dataset|table|sql|rest|file)\s+([^\s]+)\s+on\s+(.+)',
                    stripped,
                    re.IGNORECASE,
                )
                if not join_match:
                    raise self._error(
                        "Expected join syntax: join [type] dataset|table NAME on CONDITION",
                        self.pos + 1,
                        nxt,
                        hint='Valid join types: inner, left, right, full. Example: join left table customers on customer_id'
                    )
                join_type = (join_match.group(1) or 'inner').lower()
                target_type = join_match.group(2).lower()
                target_name = join_match.group(3)
                condition_text = join_match.group(4)
                condition = self._parse_expression(condition_text)
                dataset.operations.append(
                    JoinOp(
                        target_type=target_type,
                        target_name=target_name,
                        condition=condition,
                        join_type=join_type,
                    )
                )
                self._advance()
            elif lowered.startswith('transform '):
                header_line = self._advance()
                if header_line is None:
                    break
                transform = self._parse_dataset_transform_block(header_line, indent)
                dataset.transforms.append(transform)
            elif lowered.startswith('schema:'):
                block_indent = indent
                self._advance()
                fields = self._parse_dataset_schema_block(block_indent)
                dataset.schema.extend(fields)
            elif lowered.startswith('feature '):
                header_line = self._advance()
                if header_line is None:
                    break
                feature = self._parse_dataset_feature(header_line, indent)
                dataset.features.append(feature)
            elif lowered.startswith('target '):
                header_line = self._advance()
                if header_line is None:
                    break
                target = self._parse_dataset_target(header_line, indent)
                dataset.targets.append(target)
            elif lowered.startswith('quality '):
                header_line = self._advance()
                if header_line is None:
                    break
                quality = self._parse_dataset_quality_check(header_line, indent)
                dataset.quality_checks.append(quality)
            elif lowered.startswith('profile:'):
                block_indent = indent
                self._advance()
                profile_data = self._parse_kv_block(block_indent)
                dataset.profile = self._build_dataset_profile(profile_data)
            elif lowered.startswith('metadata:'):
                block_indent = indent
                self._advance()
                metadata = self._parse_kv_block(block_indent)
                dataset.metadata.update(metadata)
            elif lowered.startswith('lineage:'):
                block_indent = indent
                self._advance()
                lineage = self._parse_kv_block(block_indent)
                dataset.lineage.update(lineage)
            elif lowered.startswith('tags:'):
                tag_text = stripped[len('tags:'):].strip()
                dataset.tags = self._parse_tag_list(tag_text)
                self._advance()
            elif lowered.startswith('reactive:'):
                dataset.reactive = self._parse_bool(stripped.split(':', 1)[1])
                self._advance()
            elif lowered.startswith('auto refresh'):
                refresh_text = stripped.split('auto refresh', 1)[1].strip()
                match_refresh = re.match(
                    r'(?:every\s+)?(\d+)\s*(seconds|second|minutes|minute|ms|milliseconds)?',
                    refresh_text,
                    re.IGNORECASE,
                )
                if not match_refresh:
                    raise self._error(
                        "Expected: auto refresh every <number> [seconds|minutes|ms]",
                        self.pos + 1,
                        nxt,
                        hint='Specify refresh interval like: auto refresh every 5 minutes'
                    )
                value = int(match_refresh.group(1))
                unit = (match_refresh.group(2) or 'seconds').lower()
                interval_seconds = value
                if unit.startswith('minute'):
                    interval_seconds = value * 60
                elif unit in {'ms', 'millisecond', 'milliseconds'}:
                    interval_seconds = max(1, value // 1000)
                dataset.refresh_policy = RefreshPolicy(
                    interval_seconds=interval_seconds,
                    mode='polling',
                )
                self._advance()
            elif lowered.startswith('with option '):
                option_text = stripped[len('with option '):].strip()
                key, value = self._parse_connector_option(option_text, self.pos + 1, nxt)
                self._apply_connector_option(dataset, key, value)
                self._advance()
            elif lowered.startswith('cache:'):
                block_indent = indent
                self._advance()
                config = self._parse_kv_block(block_indent)
                dataset.cache_policy = self._build_cache_policy(config)
            elif lowered.startswith('pagination:'):
                block_indent = indent
                self._advance()
                config = self._parse_kv_block(block_indent)
                dataset.pagination = self._build_pagination_policy(config)
            elif lowered.startswith('stream'):
                block_indent = indent
                self._advance()
                config = self._parse_kv_block(block_indent)
                dataset.streaming = self._build_streaming_policy(config)
            else:
                raise self._error(
                    "Expected dataset operation or configuration block",
                    self.pos + 1,
                    nxt,
                    hint='Valid directives: filter by, add column, group by, order by, join, transform, schema, feature, target, quality, profile, metadata, lineage, tags, reactive, auto refresh, with option, cache, pagination, stream'
                )

        return dataset

    def _parse_dataset_transform_block(self, header_line: str, header_indent: int) -> DatasetTransformStep:
        """
        Parse dataset transformation specification.
        
        Transforms define data processing steps with inputs, outputs,
        expressions, and configuration options.
        """
        stripped = header_line.strip()
        has_block = stripped.endswith(':')
        header = stripped[:-1] if has_block else stripped
        try:
            tokens = shlex.split(header)
        except ValueError as exc:
            raise self._error(
                f"Unable to parse transform declaration: {exc}",
                self.pos,
                header_line,
                hint='Check for unmatched quotes in transform name'
            )
        if len(tokens) < 2 or tokens[0].lower() != 'transform':
            raise self._error(
                "Expected: transform \"Name\"",
                self.pos,
                header_line,
                hint='Transforms require a name, e.g., transform "normalize_data":'
            )

        name = self._strip_quotes(tokens[1])
        transform_type = 'custom'
        output: Optional[str] = None

        idx = 2
        while idx < len(tokens):
            token_lower = tokens[idx].lower()
            if token_lower == 'type' and idx + 1 < len(tokens):
                transform_type = tokens[idx + 1]
                idx += 2
                continue
            if token_lower == 'output' and idx + 1 < len(tokens):
                output = tokens[idx + 1]
                idx += 2
                continue
            idx += 1

        config: Dict[str, Any] = {}
        if has_block:
            config = self._parse_kv_block(header_indent)

        inputs_raw = config.pop('inputs', config.pop('input', []))
        inputs = self._ensure_string_list(inputs_raw)
        output_raw = config.pop('output', None)
        if isinstance(output_raw, str) and output_raw:
            output = output_raw
        expression_raw = config.pop('expression', config.pop('expr', None))
        expression = self._coerce_expression(expression_raw) if expression_raw is not None else None
        options = self._coerce_options_dict(config.pop('options', {}))
        if config:
            options.update(config)

        return DatasetTransformStep(
            name=name,
            transform_type=transform_type,
            inputs=inputs,
            output=output,
            expression=expression,
            options=options,
        )

    def _parse_dataset_schema_block(self, parent_indent: int) -> List[DatasetSchemaField]:
        """
        Parse dataset schema block with column definitions.
        
        Schema defines column names, types, nullability, descriptions,
        constraints, and statistics.
        """
        fields: List[DatasetSchemaField] = []
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

            name_part, _, remainder = stripped.partition(':')
            name_token = name_part.strip()
            if not name_token:
                raise self._error(
                    "Column name required inside schema block",
                    self.pos + 1,
                    line,
                    hint='Each schema field must have a name, e.g., column_name: type'
                )
            if name_token.lower().startswith('column '):
                name_token = name_token[7:]
            name = self._strip_quotes(name_token.strip())

            self._advance()

            next_line = self._peek()
            has_block = False
            if next_line is not None and self._indent(next_line) > indent:
                has_block = True

            dtype = remainder.strip() or None
            config: Dict[str, Any] = {}
            if has_block:
                config = self._parse_kv_block(indent)
            if 'dtype' in config:
                dtype_value = config.pop('dtype')
                dtype = str(dtype_value) if dtype_value is not None else dtype
            if dtype is None:
                dtype = 'any'

            nullable = self._to_bool(config.pop('nullable', True))
            description_raw = config.pop('description', config.pop('desc', None))
            description = str(description_raw) if description_raw is not None else None
            tags_raw = config.pop('tags', [])
            tags = self._ensure_string_list(tags_raw)
            constraints_raw = config.pop('constraints', {})
            constraints = self._coerce_options_dict(constraints_raw)
            stats_raw = config.pop('stats', {})
            stats = self._coerce_options_dict(stats_raw)
            if config:
                constraints.update(config)

            fields.append(
                DatasetSchemaField(
                    name=name,
                    dtype=str(dtype),
                    nullable=nullable,
                    description=description,
                    tags=tags,
                    constraints=constraints,
                    stats=stats,
                )
            )
        return fields

    def _parse_dataset_feature(self, header_line: str, header_indent: int) -> DatasetFeature:
        """
        Parse dataset feature specification for ML.
        
        Features define ML input variables with roles, types, sources,
        transformations, and descriptions.
        """
        stripped = header_line.strip()
        has_block = stripped.endswith(':')
        header = stripped[:-1] if has_block else stripped
        try:
            tokens = shlex.split(header)
        except ValueError as exc:
            raise self._error(
                f"Unable to parse feature declaration: {exc}",
                self.pos,
                header_line,
                hint='Check for unmatched quotes in feature name'
            )
        if len(tokens) < 2 or tokens[0].lower() != 'feature':
            raise self._error(
                "Expected: feature \"Name\"",
                self.pos,
                header_line,
                hint='Features require a name, e.g., feature "revenue_log":'
            )

        name = self._strip_quotes(tokens[1])
        config: Dict[str, Any] = {}
        if has_block:
            config = self._parse_kv_block(header_indent)

        role_raw = config.pop('role', 'feature')
        role = str(role_raw) if role_raw is not None else 'feature'
        source_raw = config.pop('source', config.pop('from', None))
        source = str(source_raw) if source_raw is not None else None
        dtype_raw = config.pop('dtype', None)
        dtype = str(dtype_raw) if dtype_raw is not None else None
        expr_raw = config.pop('expression', config.pop('expr', None))
        expression = self._coerce_expression(expr_raw) if expr_raw is not None else None
        description_raw = config.pop('description', config.pop('desc', None))
        description = str(description_raw) if description_raw is not None else None
        options = self._coerce_options_dict(config.pop('options', {}))
        if config:
            options.update(config)

        return DatasetFeature(
            name=name,
            source=source,
            role=role,
            dtype=dtype,
            expression=expression,
            description=description,
            options=options,
        )

    def _parse_dataset_target(self, header_line: str, header_indent: int) -> DatasetTarget:
        """
        Parse dataset target specification for ML.
        
        Targets define prediction objectives for ML models including
        classification, regression, and ranking tasks.
        """
        stripped = header_line.strip()
        has_block = stripped.endswith(':')
        header = stripped[:-1] if has_block else stripped
        try:
            tokens = shlex.split(header)
        except ValueError as exc:
            raise self._error(
                f"Unable to parse target declaration: {exc}",
                self.pos,
                header_line,
                hint='Check for unmatched quotes in target name'
            )
        if len(tokens) < 2 or tokens[0].lower() != 'target':
            raise self._error(
                "Expected: target \"Name\"",
                self.pos,
                header_line,
                hint='Targets require a name, e.g., target "conversion":'
            )

        name = self._strip_quotes(tokens[1])
        config: Dict[str, Any] = {}
        if has_block:
            config = self._parse_kv_block(header_indent)

        kind_raw = config.pop('kind', 'classification')
        kind = str(kind_raw) if kind_raw is not None else 'classification'
        expr_raw = config.pop('expression', config.pop('expr', None))
        expression = self._coerce_expression(expr_raw) if expr_raw is not None else None
        positive_raw = config.pop('positive_class', config.pop('positive', None))
        positive_class = str(positive_raw) if positive_raw is not None else None
        horizon_raw = config.pop('horizon', config.pop('window', None))
        horizon = self._coerce_int(horizon_raw)
        options = self._coerce_options_dict(config.pop('options', {}))
        if config:
            options.update(config)

        return DatasetTarget(
            name=name,
            kind=kind,
            expression=expression,
            positive_class=positive_class,
            horizon=horizon,
            options=options,
        )

    def _parse_dataset_quality_check(self, header_line: str, header_indent: int) -> DatasetQualityCheck:
        """
        Parse dataset quality check specification.
        
        Quality checks define validation rules with conditions, metrics,
        thresholds, severity levels, and alert messages.
        """
        stripped = header_line.strip()
        has_block = stripped.endswith(':')
        header = stripped[:-1] if has_block else stripped
        try:
            tokens = shlex.split(header)
        except ValueError as exc:
            raise self._error(
                f"Unable to parse quality check declaration: {exc}",
                self.pos,
                header_line,
                hint='Check for unmatched quotes in quality check name'
            )
        if len(tokens) < 2 or tokens[0].lower() != 'quality':
            raise self._error(
                "Expected: quality \"Name\"",
                self.pos,
                header_line,
                hint='Quality checks require a name, e.g., quality "no_nulls":'
            )

        name = self._strip_quotes(tokens[1])
        config: Dict[str, Any] = {}
        if has_block:
            config = self._parse_kv_block(header_indent)

        condition_raw = config.pop('condition', config.pop('expr', None))
        condition = self._coerce_expression(condition_raw) if condition_raw is not None else None
        metric_raw = config.pop('metric', config.pop('measure', None))
        metric = str(metric_raw) if metric_raw is not None else None
        threshold_raw = config.pop('threshold', config.pop('value', None))
        threshold = None
        if threshold_raw is not None:
            try:
                threshold = float(threshold_raw)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                threshold = None
        severity_raw = config.pop('severity', config.pop('level', 'error'))
        severity = str(severity_raw) if severity_raw is not None else 'error'
        message_raw = config.pop('message', config.pop('text', None))
        message = str(message_raw) if message_raw is not None else None
        extras = self._coerce_options_dict(config.pop('extras', {}))
        if config:
            extras.update(config)

        return DatasetQualityCheck(
            name=name,
            condition=condition,
            metric=metric,
            threshold=threshold,
            severity=severity,
            message=message,
            extras=extras,
        )

    def _build_dataset_profile(self, data: Dict[str, Any]) -> DatasetProfile:
        if not isinstance(data, dict):
            return DatasetProfile()
        profile_data = dict(data)
        row_count = self._coerce_int(profile_data.pop('row_count', profile_data.pop('rows', None)))
        column_count = self._coerce_int(profile_data.pop('column_count', profile_data.pop('columns', None)))
        freshness_raw = profile_data.pop('freshness', profile_data.pop('recency', None))
        freshness = str(freshness_raw) if freshness_raw is not None else None
        updated_raw = profile_data.pop('updated_at', profile_data.pop('last_updated', profile_data.pop('updated', None)))
        updated_at = str(updated_raw) if updated_raw is not None else None
        stats_raw = profile_data.pop('stats', {})
        stats = self._coerce_options_dict(stats_raw)
        if profile_data:
            extras_bucket = stats.setdefault('extras', {})
            if not isinstance(extras_bucket, dict):
                extras_bucket = {}
                stats['extras'] = extras_bucket
            extras_bucket.update(profile_data)
        return DatasetProfile(
            row_count=row_count,
            column_count=column_count,
            freshness=freshness,
            updated_at=updated_at,
            stats=stats,
        )

    def _parse_tag_list(self, raw: str) -> List[str]:
        if not raw:
            return []
        tokens: List[str] = []
        try:
            split_tokens = shlex.split(raw)
        except ValueError:
            split_tokens = [raw]
        for token in split_tokens:
            parts = token.split(',')
            for part in parts:
                cleaned = part.strip()
                if cleaned:
                    tokens.append(cleaned)
        return tokens

    def _strip_quotes(self, value: str) -> str:
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        return value

    def _ensure_string_list(self, raw: Any) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, list):
            result: List[str] = []
            for item in raw:
                if item is None:
                    continue
                if isinstance(item, str):
                    result.append(item)
                else:
                    result.append(self._stringify_value(item))
            return [item for item in result if item]
        if isinstance(raw, str):
            if ',' in raw:
                tokens = [part.strip() for part in raw.split(',') if part.strip()]
            else:
                tokens = [part.strip() for part in raw.split() if part.strip()]
            return tokens
        return [self._stringify_value(raw)]

    def _coerce_options_dict(self, raw: Any) -> Dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        return {"value": raw}

    def _to_bool(self, value: Any, default: bool = True) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {'true', 'yes', '1', 'on'}:
                return True
            if lowered in {'false', 'no', '0', 'off'}:
                return False
        return bool(value)

    def _apply_connector_option(self, dataset: Dataset, key: str, value: Any) -> None:
        if dataset.connector is None:
            dataset.connector = DatasetConnectorConfig(
                connector_type=dataset.source_type,
                connector_name=dataset.source,
            )
        options = dataset.connector.options
        parts = [part.strip() for part in key.split('.') if part.strip()]
        if not parts:
            raise self._error("Connector option key cannot be empty", self.pos + 1)
        target = options
        for part in parts[:-1]:
            existing = target.get(part)
            if not isinstance(existing, dict):
                existing = {}
                target[part] = existing
            target = existing
        target[parts[-1]] = value

    def _parse_connector_option(self, text: str, line_no: int, line: str) -> tuple[str, Any]:
        if not text:
            raise self._error(
                "Connector option requires 'with option key value'",
                line_no,
                line,
                hint='Specify options like: with option timeout 30 or with option retry.max 3'
            )
        key, _, value_text = text.partition(' ')
        key = key.strip()
        if not key:
            raise self._error(
                "Connector option key is missing",
                line_no,
                line,
                hint='Provide a configuration key, e.g., with option batch_size 100'
            )
        value_text = value_text.strip()
        if not value_text:
            value: Any = True
        else:
            value = self._coerce_connector_option_value(value_text)
        return key, value

    def _coerce_connector_option_value(self, raw: str) -> Any:
        raw = raw.strip()
        if not raw:
            return ""
        context_ref = self._parse_context_reference(raw)
        if context_ref is not None:
            return context_ref
        if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, str):
                    context_ref_inner = self._parse_context_reference(parsed)
                    if context_ref_inner is not None:
                        return context_ref_inner
                return parsed
            except (SyntaxError, ValueError):
                inner = raw[1:-1]
                context_ref_inner = self._parse_context_reference(inner)
                if context_ref_inner is not None:
                    return context_ref_inner
                return inner
        lowered = raw.lower()
        if lowered in {'true', 'false'}:
            return lowered == 'true'
        if lowered in {'null', 'none'}:
            return None
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw
