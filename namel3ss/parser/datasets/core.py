"""Core dataset parsing logic."""

from __future__ import annotations

import re
import shlex
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...ast import Dataset, DatasetConnectorConfig


class CoreDatasetParserMixin:
    """Mixin for core dataset parsing logic - header and operations."""
    
    def _parse_dataset(self, line: str, line_no: int, base_indent: int) -> "Dataset":
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
        # Parse header and create dataset
        dataset = self._parse_dataset_header(line, line_no, base_indent)
        
        # Parse operations and configurations
        self._parse_dataset_body(dataset, base_indent, line_no)
        
        return dataset
    
    def _parse_dataset_header(self, line: str, line_no: int, base_indent: int) -> "Dataset":
        """
        Parse dataset declaration header (name and connector config).
        
        Returns initialized Dataset with connector configured.
        """
        from ...ast import Dataset, DatasetConnectorConfig
        
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

        # Parse connector configuration based on source type
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
            options: dict = {}
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
            options: dict = {}
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

        return Dataset(name=name, source_type=source_type, source=source, connector=connector)
    
    def _parse_dataset_body(self, dataset: "Dataset", base_indent: int, line_no: int) -> None:
        """
        Parse dataset body (operations, schema, features, etc.).
        
        Modifies dataset in place with parsed configurations.
        """
        from ...ast import (
            FilterOp, ComputedColumnOp, WindowOp, GroupByOp, OrderByOp,
            AggregateOp, JoinOp, RefreshPolicy
        )
        
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
            
            # Filter operations
            if lowered.startswith('filter by:'):
                cond_text = stripped[len('filter by:'):].strip()
                condition = self._parse_expression(cond_text)
                dataset.operations.append(FilterOp(condition=condition))
                self._advance()
            
            # Computed columns
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

                # Check for window functions
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
            
            # Group by
            elif lowered.startswith('group by:'):
                cols = stripped[len('group by:'):].strip()
                col_list = [c.strip() for c in cols.split(',') if c.strip()]
                dataset.operations.append(GroupByOp(columns=col_list))
                self._advance()
            
            # Order by
            elif lowered.startswith('order by:'):
                cols = stripped[len('order by:'):].strip()
                order_cols = [c.strip() for c in cols.split(',') if c.strip()]
                dataset.operations.append(OrderByOp(columns=order_cols))
                self._advance()
            
            # Aggregations
            elif re.match(r'(sum|count|avg|min|max):', lowered):
                func, expr = stripped.split(':', 1)
                expr = expr.strip()
                dataset.operations.append(
                    AggregateOp(function=func.strip(), expression=expr)
                )
                self._advance()
            
            # Joins
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
            
            # Transform blocks
            elif lowered.startswith('transform '):
                header_line = self._advance()
                if header_line is None:
                    break
                steps = self._parse_dataset_transform_block(indent)
                dataset.transforms.extend(steps)
            
            # Schema blocks
            elif lowered.startswith('schema:'):
                block_indent = indent
                self._advance()
                fields = self._parse_dataset_schema_block(block_indent)
                dataset.schema.extend(fields)
            
            # Feature specifications
            elif lowered.startswith('feature '):
                header_line = self._advance()
                if header_line is None:
                    break
                feature = self._parse_dataset_feature(header_line, indent)
                dataset.features.append(feature)
            
            # Target specifications
            elif lowered.startswith('target '):
                header_line = self._advance()
                if header_line is None:
                    break
                target = self._parse_dataset_target(header_line, indent)
                dataset.targets.append(target)
            
            # Quality checks
            elif lowered.startswith('quality '):
                header_line = self._advance()
                if header_line is None:
                    break
                quality = self._parse_dataset_quality_check(header_line, indent)
                dataset.quality_checks.append(quality)
            
            # Profile block
            elif lowered.startswith('profile:'):
                block_indent = indent
                self._advance()
                profile_data = self._parse_kv_block(block_indent)
                dataset.profile = self._build_dataset_profile(profile_data)
            
            # Metadata block
            elif lowered.startswith('metadata:'):
                block_indent = indent
                self._advance()
                metadata = self._parse_kv_block(block_indent)
                dataset.metadata.update(metadata)
            
            # Lineage block
            elif lowered.startswith('lineage:'):
                block_indent = indent
                self._advance()
                lineage = self._parse_kv_block(block_indent)
                dataset.lineage.update(lineage)
            
            # Tags
            elif lowered.startswith('tags:'):
                tag_text = stripped[len('tags:'):].strip()
                dataset.tags = self._parse_tag_list(tag_text)
                self._advance()
            
            # Reactive flag
            elif lowered.startswith('reactive:'):
                dataset.reactive = self._parse_bool(stripped.split(':', 1)[1])
                self._advance()
            
            # Auto refresh
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
            
            # Connector options
            elif lowered.startswith('with option '):
                option_text = stripped[len('with option '):].strip()
                key, value = self._parse_connector_option(option_text, self.pos + 1, nxt)
                self._apply_connector_option(dataset, key, value)
                self._advance()
            
            # Cache policy
            elif lowered.startswith('cache:'):
                block_indent = indent
                self._advance()
                config = self._parse_kv_block(block_indent)
                dataset.cache_policy = self._build_cache_policy(config)
            
            # Pagination policy
            elif lowered.startswith('pagination:'):
                block_indent = indent
                self._advance()
                config = self._parse_kv_block(block_indent)
                dataset.pagination = self._build_pagination_policy(config)
            
            # Streaming policy
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
