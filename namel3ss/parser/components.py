from __future__ import annotations

import re
import shlex
from typing import Any, Dict, List, Optional, Union

from namel3ss.ast.design_tokens import (
    validate_variant,
    validate_tone,
    validate_density,
    validate_size,
)
from namel3ss.ast import (
    AccordionItem,
    AccordionLayout,
    ActionOperationType,
    AvatarItem,
    BadgeConfig,
    BreadcrumbItem,
    Breadcrumbs,
    CallPythonOperation,
    CardFooter,
    CardHeader,
    CardItemConfig,
    CardSection,
    ChartConfig,
    ColumnConfig,
    CommandPalette,
    CommandSource,
    ConditionalAction,
    ContextValue,
    EmptyStateConfig,
    FieldValueConfig,
    FormField,
    GridLayout,
    InfoGridItem,
    InferenceTarget,
    LayoutMeta,
    LayoutSpec,
    ListItemConfig,
    Modal,
    ModalAction,
    Navbar,
    NavbarAction,
    NavItem,
    NavSection,
    PageStatement,
    PredictStatement,
    ShowAvatarGroup,
    ShowCard,
    ShowChart,
    ShowDataChart,
    ShowDataList,
    ShowDataTable,
    ShowForm,
    ShowList,
    ShowStatSummary,
    ShowTable,
    ShowText,
    ShowTimeline,
    Sidebar,
    SparklineConfig,
    SplitLayout,
    StackLayout,
    Statement,
    TabItem,
    TabsLayout,
    TimelineItem,
    Toast,
    ToolbarConfig,
    VariableAssignment,
)

from .actions import ActionParserMixin
from .base import N3SyntaxError
# KeywordRegistry import removed - class does not exist


class ComponentParserMixin(ActionParserMixin):
    """
    Mixin for parsing UI component statements within pages.
    
    This parser handles visual display components including text displays,
    tables, charts, forms, predictions, and variable assignments. These
    components define the interactive elements and data visualizations
    presented to users on N3 pages.
    
    Syntax Example:
        show text "Welcome to Dashboard":
            color: blue
            font size: 18px
        
        show table "Sales Data" from dataset sales:
            columns: product, quantity, revenue
            filter by: quantity > 0
            sort by: revenue desc
        
        show chart "Revenue Trend" from table sales:
            type: line
            x: month
            y: revenue
            color: region
        
        show form "Add Product":
            fields: name:text, price:number, category:select
            on submit:
                create record in table products
        
        predict using model sales_forecast with dataset recent_sales into variable forecast
    
    Features:
        - Text displays with styling options
        - Data tables with filtering, sorting, column selection
        - Charts with multiple types (bar, line, pie, scatter) and configurations
        - Forms with field definitions and submit actions
        - ML predictions with model inference
        - Variable assignments for data manipulation
    
    Supported Components:
        - show text: Display text with styling
        - show table: Tabular data display
        - show chart: Data visualizations
        - show form: Input forms with validation
        - predict: ML model inference
        - set: Variable assignment
    """

    def _parse_show_text(self, line: str, base_indent: int) -> ShowText:
        """
        Parse a text display component.
        
        Syntax: show text "Message"
        
        Optional styling properties can be specified in indented block.
        """
        match = re.match(r'show\s+text\s+"([^"]+)"(?:\s+(.*))?$', line.strip())
        if not match:
            raise self._error(
                "Expected: show text \"Message\"",
                self.pos,
                line,
                hint='Text components require a message in quotes, e.g., show text "Hello World"'
            )
        text = match.group(1)
        inline_styles = match.group(2)
        styles: Dict[str, str] = {}

        if inline_styles:
            inline_match = re.match(r'([\w_]+):\s*(.+)', inline_styles)
            if inline_match:
                styles[inline_match.group(1)] = inline_match.group(2).strip()
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
                self.pos,
                context="show text style properties",
                hint="Style properties must be indented under the show text declaration"
            )
            match_style = re.match(r'([\w\s]+):\s*(.+)', stripped)
            if not match_style:
                break
            key = match_style.group(1).strip()
            value = match_style.group(2).strip()
            styles[key] = value
            self._advance()
        return ShowText(text=text, styles=styles)

    def _parse_show_table(self, line: str, base_indent: int) -> ShowTable:
        """
        Parse a table display component.
        
        Syntax: 
          show table "Title" from table|dataset|frame|model SOURCE
          show table "Title" from table|dataset|frame|model SOURCE (token=value, ...)
          show table from model "SOURCE" (token=value, ...)  # title auto-generated from model
        
        Supports columns, filtering, sorting, styling, and layout configuration.
        """
        # Try to match with title first
        match = re.match(
            r'show\s+table\s+"([^\"]+)"\s+from\s+(table|dataset|frame|model)\s+"?([^\s\("]+)"?\s*(?:\(([^)]+)\))?\s*$',
            line.strip(),
        )
        
        # Try without title (model-based syntax)
        if not match:
            match_notitle = re.match(
                r'show\s+table\s+from\s+model\s+"([^\"]+)"\s*(?:\(([^)]+)\))?\s*$',
                line.strip(),
            )
            if match_notitle:
                source = match_notitle.group(1)
                title = source.capitalize() + "s"  # Auto-generate title (e.g., "User" -> "Users")
                source_type = "model"
                inline_tokens_str = match_notitle.group(2) if match_notitle.group(2) else None
            else:
                raise self._error(
                    "Expected: show table \"Title\" from table|dataset|frame|model SOURCE or show table from model \"SOURCE\"",
                    self.pos,
                    line,
                    hint='Table components require a title and data source, e.g., show table "Sales" from dataset sales'
                )
        else:
            title = match.group(1)
            source_type = match.group(2)
            source = match.group(3)
            inline_tokens_str = match.group(4) if match.group(4) else None
        
        # Parse inline tokens if present
        variant = None
        tone = None
        density = None
        size = None
        theme = None
        color_scheme = None
        
        if inline_tokens_str:
            # Parse comma-separated token=value pairs
            for token_pair in inline_tokens_str.split(','):
                token_pair = token_pair.strip()
                if '=' in token_pair:
                    key, value = token_pair.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == 'variant':
                        variant = self._parse_design_token(value, 'variant', self.pos + 1, line)
                    elif key == 'tone':
                        tone = self._parse_design_token(value, 'tone', self.pos + 1, line)
                    elif key == 'density':
                        density = self._parse_design_token(value, 'density', self.pos + 1, line)
                    elif key == 'size':
                        size = self._parse_design_token(value, 'size', self.pos + 1, line)
                    elif key == 'theme':
                        theme = self._parse_design_token(value, 'theme', self.pos + 1, line)
                    elif key == 'color_scheme':
                        color_scheme = self._parse_design_token(value, 'color_scheme', self.pos + 1, line)
        
        columns: Optional[List[str]] = None
        filter_by: Optional[str] = None
        sort_by: Optional[str] = None
        style_values: Dict[str, Any] = {}
        style_block: Optional[Dict[str, Any]] = None
        layout_meta: Optional[LayoutMeta] = None
        insight_name: Optional[str] = None
        
        # Continue parsing block-style tokens (can override inline tokens)
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
            if stripped.startswith('columns:'):
                cols = stripped[len('columns:'):].strip()
                columns = [c.strip() for c in cols.split(',') if c.strip()]
                self._advance()
            elif stripped.startswith('filter by:'):
                filter_by = stripped[len('filter by:'):].strip()
                self._advance()
            elif stripped.startswith('sort by:'):
                sort_by = stripped[len('sort by:'):].strip()
                self._advance()
            elif stripped.startswith('style:'):
                block_indent = indent
                self._advance()
                style_block = self._parse_kv_block(block_indent)
            elif stripped.startswith('layout:'):
                block_indent = indent
                self._advance()
                block = self._parse_kv_block(block_indent)
                layout_meta = self._build_layout_meta(block)
            elif stripped.startswith('insight:'):
                insight_raw = stripped[len('insight:'):].strip()
                value = self._coerce_scalar(insight_raw) if insight_raw else None
                if isinstance(value, ContextValue):
                    insight_name = f"{value.scope}:{'.'.join(value.path)}"
                elif value is not None:
                    insight_name = str(value)
                else:
                    insight_name = None
                self._advance()
            elif stripped.startswith('variant:'):
                value = stripped[len('variant:'):].strip()
                variant = self._parse_design_token(value, "variant", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('tone:'):
                value = stripped[len('tone:'):].strip()
                tone = self._parse_design_token(value, "tone", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('density:'):
                value = stripped[len('density:'):].strip()
                density = self._parse_design_token(value, "density", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('size:'):
                value = stripped[len('size:'):].strip()
                size = self._parse_design_token(value, "size", self.pos + 1, nxt)
                self._advance()
            else:
                match_style = re.match(r'([\w\s]+):\s*(.+)', stripped)
                if not match_style:
                    raise self._error(
                        "Expected table property ('columns:', 'filter by:', 'sort by:' or style)",
                        self.pos + 1,
                        nxt,
                        hint='Valid table properties: columns, filter by, sort by, style, layout, insight, variant, tone, density, size'
                    )
                key = match_style.group(1).strip()
                value = self._coerce_scalar(match_style.group(2).strip())
                style_values[key] = value
                self._advance()
        combined_style: Dict[str, Any] = {}
        if style_block:
            combined_style.update(style_block)
        if style_values:
            combined_style.update(style_values)
        style_payload = combined_style or None
        return ShowTable(
            title=title,
            source_type=source_type,
            source=source,
            columns=columns,
            filter_by=filter_by,
            sort_by=sort_by,
            style=style_payload,
            layout=layout_meta,
            insight=insight_name,
            variant=variant,
            tone=tone,
            density=density,
            size=size,
            theme=theme,
            color_scheme=color_scheme,
        )

    def _parse_show_chart(self, line: str, base_indent: int) -> ShowChart:
        """
        Parse a chart display component.
        
        Syntax: show chart "Title" from table|dataset|frame|file SOURCE
        
        Supports chart types (bar, line, pie, scatter), axes mapping,
        styling, legends, and layout configuration.
        """
        match = re.match(
            r'show\s+chart\s+"([^\"]+)"\s+from\s+(table|dataset|frame|file)\s+([^\s]+)\s*$',
            line.strip(),
        )
        if not match:
            raise self._error(
                "Expected: show chart \"Title\" from table|dataset|frame|file SOURCE",
                self.pos,
                line,
                hint='Chart components require a title and data source, e.g., show chart "Revenue" from dataset sales'
            )
        heading = match.group(1)
        source_type = match.group(2)
        source = match.group(3)
        chart_type = 'bar'
        x = None
        y = None
        color = None
        style_inline: Dict[str, Any] = {}
        style_blocks: Dict[str, Any] = {}
        general_style: Dict[str, Any] = {}
        layout_meta: Optional[LayoutMeta] = None
        chart_title_value: Optional[Any] = None
        legend_config: Optional[Dict[str, Any]] = None
        insight_name: Optional[str] = None
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
            if stripped.startswith('type:'):
                chart_type = stripped[len('type:'):].strip()
                self._advance()
            elif stripped.startswith('x:'):
                x = stripped[len('x:'):].strip()
                self._advance()
            elif stripped.startswith('y:'):
                y = stripped[len('y:'):].strip()
                self._advance()
            elif stripped.startswith('color:'):
                color = stripped[len('color:'):].strip()
                self._advance()
            elif stripped.startswith('title:'):
                remainder = stripped[len('title:'):].strip()
                self._advance()
                title_meta: Dict[str, Any] = {}
                if remainder:
                    value = self._coerce_scalar(remainder)
                    if isinstance(value, dict):
                        title_meta.update(value)
                        text_value = title_meta.get('text') or title_meta.get('value')
                        if text_value is not None:
                            chart_title_value = text_value
                    else:
                        if isinstance(value, bool):
                            title_meta['show'] = value
                        else:
                            chart_title_value = value
                else:
                    title_meta = self._parse_kv_block(indent)
                    text_value = title_meta.get('text') or title_meta.get('value')
                    if text_value is not None:
                        chart_title_value = text_value
                if title_meta:
                    style_blocks.setdefault('title', {}).update(title_meta)
                elif chart_title_value is not None:
                    style_blocks.setdefault('title', {})['text'] = chart_title_value
            elif stripped.startswith('layout:'):
                block_indent = indent
                self._advance()
                block = self._parse_kv_block(block_indent)
                layout_meta = self._build_layout_meta(block)
            elif stripped.startswith('legend:'):
                remainder = stripped[len('legend:'):].strip()
                self._advance()
                if remainder:
                    value = self._coerce_scalar(remainder)
                    if isinstance(value, dict):
                        legend_config = value
                    else:
                        if isinstance(value, bool):
                            legend_config = {'show': value}
                        else:
                            legend_config = {'position': value}
                else:
                    legend_config = self._parse_kv_block(indent)
            elif stripped.startswith('colors:'):
                block_indent = indent
                self._advance()
                style_blocks['colors'] = self._parse_kv_block(block_indent)
            elif stripped.startswith('axes:'):
                block_indent = indent
                self._advance()
                style_blocks['axes'] = self._parse_kv_block(block_indent)
            elif stripped.startswith('style:'):
                block_indent = indent
                self._advance()
                general_style.update(self._parse_kv_block(block_indent))
            elif stripped.startswith('insight:'):
                insight_raw = stripped[len('insight:'):].strip()
                value = self._coerce_scalar(insight_raw) if insight_raw else None
                if isinstance(value, ContextValue):
                    insight_name = f"{value.scope}:{'.'.join(value.path)}"
                elif value is not None:
                    insight_name = str(value)
                else:
                    insight_name = None
                self._advance()
            else:
                match_style = re.match(r'([\w\s]+):\s*(.+)', stripped)
                if not match_style:
                    raise self._error(
                        "Expected chart property ('type:', 'x:', 'y:', 'color:', 'title:', 'legend:', 'colors:', 'axes:', 'style:' or layout)",
                        self.pos + 1,
                        nxt,
                        hint='Valid chart properties: type, x, y, color, title, legend, colors, axes, style, layout, insight'
                    )
                key = match_style.group(1).strip()
                value = self._coerce_scalar(match_style.group(2).strip())
                style_inline[key] = value
                self._advance()
        combined_style: Dict[str, Any] = {}
        if general_style:
            combined_style.update(general_style)
        if style_inline:
            combined_style.update(style_inline)
        if style_blocks:
            for key, value in style_blocks.items():
                combined_style[key] = value
        style_payload = combined_style or None
        return ShowChart(
            heading=heading,
            source_type=source_type,
            source=source,
            chart_type=chart_type,
            x=x,
            y=y,
            color=color,
            layout=layout_meta,
            insight=insight_name,
            style=style_payload,
            title=chart_title_value,
            legend=legend_config,
        )

    def _parse_show_form(self, line: str, base_indent: int) -> ShowForm:
        """
        Parse a production-grade form component.
        
        Syntax: 
            show form "Title" [(variant=X, tone=Y, size=Z)]:
                fields: [...]
                layout: vertical|horizontal|inline
                submit_action: action_name
                initial_values_binding: dataset_name
                
        Supports semantic field components with validation, bindings, and conditional rendering.
        """
        # Updated regex to handle optional design tokens in parentheses
        match = re.match(r'show\s+form\s+"([^\"]+)"\s*(\([^)]*\))?\s*:?', line.strip())
        if not match:
            raise self._error(
                "Expected: show form \"Title\" [(design tokens)]:",
                self.pos,
                line,
                hint='Form components require a title in quotes, e.g., show form "User Registration" (variant=outlined):'
            )
        title = match.group(1)
        tokens_str = match.group(2)  # e.g., "(variant=outlined, tone=success)"
        
        # Parse design tokens from header if present
        variant = None
        tone = None
        density = None
        size = None
        theme = None
        color_scheme = None
        
        if tokens_str:
            # Remove parentheses and parse key=value pairs
            tokens_str = tokens_str.strip('()').strip()
            for token_pair in tokens_str.split(','):
                token_pair = token_pair.strip()
                if '=' not in token_pair:
                    continue
                key, value = token_pair.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'variant':
                    variant = self._parse_design_token(value, 'variant', self.pos + 1, line)
                elif key == 'tone':
                    tone = self._parse_design_token(value, 'tone', self.pos + 1, line)
                elif key == 'density':
                    density = self._parse_design_token(value, 'density', self.pos + 1, line)
                elif key == 'size':
                    size = self._parse_design_token(value, 'size', self.pos + 1, line)
                elif key == 'theme':
                    theme = self._parse_design_token(value, 'theme', self.pos + 1, line)
                elif key == 'color_scheme':
                    color_scheme = self._parse_design_token(value, 'color_scheme', self.pos + 1, line)
        
        fields: List[FormField] = []
        on_submit_ops: List[ActionOperationType] = []
        styles: Dict[str, str] = {}
        layout_spec = LayoutSpec()
        layout_mode = "vertical"
        submit_action = None
        initial_values_binding = None
        bound_dataset = None
        bound_record_id = None
        validation_mode = "on_blur"
        submit_button_text = None
        reset_button = False
        loading_text = None
        success_message = None
        error_message = None
        
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
                
            # Parse legacy inline fields (fields: name, email, message)
            if stripped.startswith('fields:'):
                rest = stripped[len('fields:'):].strip()
                if rest and ',' in rest:  # Inline comma-separated field list
                    field_parts = [p.strip() for p in rest.split(',') if p.strip()]
                    for fp in field_parts:
                        if ':' in fp:
                            fname, ftype = [part.strip() for part in fp.split(':', 1)]
                            fields.append(FormField(name=fname, component=ftype, field_type=ftype))
                        else:
                            fields.append(FormField(name=fp))
                    self._advance()
                    continue
                # Otherwise, it's declarative array syntax on following lines
                else:
                    self._advance()
                    fields = self._parse_form_fields_block(indent)
                    continue
            # Parse layout mode
            elif stripped.startswith('layout:'):
                value = stripped[len('layout:'):].strip()
                if value in ('vertical', 'horizontal', 'inline'):
                    layout_mode = value
                else:
                    # Legacy layout block
                    block_indent = indent
                    self._advance()
                    block = self._parse_kv_block(block_indent)
                    layout_spec = self._build_layout_spec(block)
                    continue
                self._advance()
                
            # Parse submit action
            elif stripped.startswith('submit_action:'):
                submit_action = stripped[len('submit_action:'):].strip()
                self._advance()
                
            # Parse initial values binding
            elif stripped.startswith('initial_values_binding:'):
                initial_values_binding = stripped[len('initial_values_binding:'):].strip()
                self._advance()
                
            # Parse bound dataset (legacy)
            elif stripped.startswith('bound_dataset:'):
                bound_dataset = stripped[len('bound_dataset:'):].strip()
                self._advance()
                
            # Parse validation mode
            elif stripped.startswith('validation_mode:'):
                validation_mode = stripped[len('validation_mode:'):].strip()
                self._advance()
                
            # Parse submit button text
            elif stripped.startswith('submit_button_text:'):
                value = stripped[len('submit_button_text:'):].strip()
                submit_button_text = value.strip('"\'')
                self._advance()
                
            # Parse reset button
            elif stripped.startswith('reset_button:'):
                value = stripped[len('reset_button:'):].strip().lower()
                reset_button = value in ('true', 'yes', '1')
                self._advance()
                
            # Parse messages
            elif stripped.startswith('loading_text:'):
                loading_text = stripped[len('loading_text:'):].strip().strip('"\'')
                self._advance()
            elif stripped.startswith('success_message:'):
                success_message = stripped[len('success_message:'):].strip().strip('"\'')
                self._advance()
            elif stripped.startswith('error_message:'):
                error_message = stripped[len('error_message:'):].strip().strip('"\'')
                self._advance()
            
            # Parse design tokens
            elif stripped.startswith('variant:'):
                value = stripped[len('variant:'):].strip()
                variant = self._parse_design_token(value, "variant", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('tone:'):
                value = stripped[len('tone:'):].strip()
                tone = self._parse_design_token(value, "tone", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('density:'):
                value = stripped[len('density:'):].strip()
                density = self._parse_design_token(value, "density", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('size:'):
                value = stripped[len('size:'):].strip()
                size = self._parse_design_token(value, "size", self.pos + 1, nxt)
                self._advance()
                
            # Parse legacy on submit operations
            elif stripped.startswith('on submit:'):
                op_base_indent = self._indent(nxt)
                self._advance()
                while self.pos < len(self.lines):
                    sub = self._peek()
                    if sub is None:
                        break
                    sub_indent = self._indent(sub)
                    sub_stripped = sub.strip()
                    if not sub_stripped or sub_stripped.startswith('#'):
                        self._advance()
                        continue
                    if sub_indent <= op_base_indent:
                        break
                    op = self._parse_action_operation(sub_stripped)
                    on_submit_ops.append(op)
                    
            # Parse other properties as styles
            else:
                match_style = re.match(r'([\w\s]+):\s*(.+)', stripped)
                if match_style:
                    key = match_style.group(1).strip()
                    value = match_style.group(2).strip()
                    styles[key] = value
                self._advance()
                
        return ShowForm(
            title=title,
            fields=fields,
            layout_mode=layout_mode,
            submit_action=submit_action,
            on_submit_ops=on_submit_ops,
            initial_values_binding=initial_values_binding,
            bound_dataset=bound_dataset,
            bound_record_id=bound_record_id,
            validation_mode=validation_mode,
            submit_button_text=submit_button_text,
            reset_button=reset_button,
            loading_text=loading_text,
            success_message=success_message,
            error_message=error_message,
            styles=styles,
            layout=layout_spec,
            variant=variant,
            tone=tone,
            density=density,
            size=size,
            theme=theme,
            color_scheme=color_scheme,
        )
    
    def _parse_form_fields_block(self, parent_indent: int) -> List[FormField]:
        """
        Parse a fields block with declarative field definitions.
        
        Syntax 1 (List style):
            fields:
              - name: email
                component: text_input
                label: "Email Address"
                required: true
                placeholder: "you@example.com"
              - name: role
                component: select
                options_binding: datasets.roles
                
        Syntax 2 (Inline style with tokens):
            fields:
              username: text
              email: text (size=md, tone=primary)
              password: password (variant=outlined)
        """
        fields = []
        
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
                
            # Check for list style field start (either "- name:" or "{ name:")
            if stripped.startswith('-') or stripped.startswith('{'):
                field = self._parse_form_field_definition(indent)
                if field:
                    fields.append(field)
            # Check for inline style: "fieldname: type" or "fieldname: type (tokens)"
            elif ':' in stripped and not stripped.startswith('-'):
                field = self._parse_inline_field_with_tokens(stripped, indent)
                if field:
                    fields.append(field)
                self._advance()
            else:
                break
                
        return fields
    
    def _parse_inline_field_with_tokens(self, line: str, indent: int) -> Optional[FormField]:
        """
        Parse inline field syntax: fieldname: type or fieldname: type (token=value, ...)
        
        Examples:
            username: text
            email: text (size=md, tone=primary)
            password: password (variant=outlined, size=lg)
        """
        import re
        
        # Match: fieldname: type [(token=value, ...)]
        match = re.match(r'([a-z_][a-z0-9_]*):\s*([a-z_][a-z0-9_]*)(?:\s*\(([^)]+)\))?', line.strip())
        if not match:
            return None
        
        field_name = match.group(1)
        field_type = match.group(2)
        tokens_str = match.group(3) if match.group(3) else None
        
        # Parse design tokens if present
        variant = None
        tone = None
        size = None
        density = None
        
        if tokens_str:
            for token_pair in tokens_str.split(','):
                token_pair = token_pair.strip()
                if '=' not in token_pair:
                    continue
                key, value = token_pair.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'variant':
                    variant = self._parse_design_token(value, 'variant', self.pos + 1, line)
                elif key == 'tone':
                    tone = self._parse_design_token(value, 'tone', self.pos + 1, line)
                elif key == 'size':
                    size = self._parse_design_token(value, 'size', self.pos + 1, line)
                elif key == 'density':
                    density = self._parse_design_token(value, 'density', self.pos + 1, line)
        
        return FormField(
            name=field_name,
            component=field_type,
            field_type=field_type,
            variant=variant,
            tone=tone,
            size=size,
        )
    
    def _parse_form_field_definition(self, field_indent: int) -> Optional[FormField]:
        """Parse a single field definition from declarative syntax."""
        line = self._peek()
        if not line:
            return None
            
        stripped = line.strip()
        
        # Handle inline object syntax: { name: "email", component: text_input, ... }
        if stripped.startswith('{'):
            self._advance()
            field_dict = self._parse_inline_field_object(stripped)
            return self._build_form_field_from_dict(field_dict)
            
        # Handle list syntax with properties on following lines
        if stripped.startswith('-'):
            # Remove the leading dash and check what follows
            after_dash = stripped[1:].strip()
            
            # Check if it's an inline object after the dash: - { name: "x", ... }
            if after_dash.startswith('{'):
                self._advance()
                field_dict = self._parse_inline_field_object(after_dash)
                return self._build_form_field_from_dict(field_dict)
            
            field_dict = {}
            
            if after_dash:
                # Parse property on same line as dash: "- name: email"
                match = re.match(r'([a-z_][a-z0-9_]*):\s*(.+)', after_dash)
                if match:
                    key = match.group(1)
                    value_str = match.group(2).strip()
                    field_dict[key] = self._parse_field_property_value(key, value_str)
            
            self._advance()
            # Parse additional properties on following lines
            more_props = self._parse_field_properties_block(field_indent)
            field_dict.update(more_props)
            return self._build_form_field_from_dict(field_dict)
            
        return None
    
    def _parse_inline_field_object(self, line: str) -> Dict[str, Any]:
        """Parse inline field object: { name: "email", component: text_input, required: true }"""
        # Simple parser for inline object notation
        field_dict = {}
        content = line.strip('{}').strip()
        
        # Split by commas, but respect quoted strings
        parts = []
        current = ""
        in_quotes = False
        for char in content:
            if char == '"' or char == "'":
                in_quotes = not in_quotes
            if char == ',' and not in_quotes:
                parts.append(current.strip())
                current = ""
            else:
                current += char
        if current.strip():
            parts.append(current.strip())
            
        for part in parts:
            if ':' in part:
                key, value = part.split(':', 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                
                # Convert boolean strings
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                # Convert numeric strings
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
                    
                field_dict[key] = value
                
        return field_dict
    
    def _parse_field_properties_block(self, field_indent: int) -> Dict[str, Any]:
        """Parse field properties from a multi-line block."""
        field_dict = {}
        
        while self.pos < len(self.lines):
            line = self._peek()
            if line is None:
                break
            indent = self._indent(line)
            stripped = line.strip()
            
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= field_indent:
                break
                
            # Parse key: value pair
            match = re.match(r'([a-z_][a-z0-9_]*):\s*(.+)', stripped)
            if match:
                key = match.group(1)
                value_str = match.group(2).strip()
                
                # Parse value based on type
                value = self._parse_field_property_value(key, value_str)
                field_dict[key] = value
                self._advance()
            else:
                break
                
        return field_dict
    
    def _parse_field_property_value(self, key: str, value_str: str) -> Any:
        """Parse and convert field property value to appropriate type."""
        # Remove quotes
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]
            
        # Boolean
        if value_str.lower() in ('true', 'false'):
            return value_str.lower() == 'true'
            
        # Integer
        if value_str.isdigit():
            return int(value_str)
            
        # Float
        if value_str.replace('.', '', 1).isdigit() and value_str.count('.') == 1:
            return float(value_str)
            
        # Expression (starts with variable or function call)
        if key in ('default', 'initial_value', 'disabled', 'visible', 'bound_record_id'):
            # Parse as expression (simplified - real implementation would use expression parser)
            return value_str
            
        return value_str
    
    def _build_form_field_from_dict(self, field_dict: Dict[str, Any]) -> FormField:
        """Build FormField instance from parsed dictionary."""
        if 'name' not in field_dict:
            raise ValueError("Field must have a 'name' property")
        
        # Parse design tokens if present
        variant = None
        tone = None
        size = None
        
        if 'variant' in field_dict:
            variant = self._parse_design_token(str(field_dict['variant']), "variant", self.pos, "field definition")
        if 'tone' in field_dict:
            tone = self._parse_design_token(str(field_dict['tone']), "tone", self.pos, "field definition")
        if 'size' in field_dict:
            size = self._parse_design_token(str(field_dict['size']), "size", self.pos, "field definition")
            
        return FormField(
            name=field_dict.get('name'),
            component=field_dict.get('component', 'text_input'),
            label=field_dict.get('label'),
            placeholder=field_dict.get('placeholder'),
            help_text=field_dict.get('help_text'),
            required=field_dict.get('required', False),
            default=field_dict.get('default'),
            initial_value=field_dict.get('initial_value'),
            min_length=field_dict.get('min_length'),
            max_length=field_dict.get('max_length'),
            pattern=field_dict.get('pattern'),
            min_value=field_dict.get('min_value'),
            max_value=field_dict.get('max_value'),
            step=field_dict.get('step'),
            options_binding=field_dict.get('options_binding'),
            options=field_dict.get('options', []),
            disabled=field_dict.get('disabled'),
            visible=field_dict.get('visible'),
            multiple=field_dict.get('multiple', False),
            accept=field_dict.get('accept'),
            max_file_size=field_dict.get('max_file_size'),
            upload_endpoint=field_dict.get('upload_endpoint'),
            field_type=field_dict.get('field_type'),  # Backward compatibility
            variant=variant,
            tone=tone,
            size=size,
        )

    def _parse_show_card(self, line: str, base_indent: int) -> ShowCard:
        """
        Parse a card display component.
        
        Syntax: show card "Title" from dataset|table|frame SOURCE
        
        Supports empty_state, item config with header/sections/actions/footer,
        group_by, filter_by, sort_by, and layout configuration.
        """
        match = re.match(
            r'show\s+card\s+"([^\"]+)"\s+from\s+(dataset|table|frame)\s+([^\s:]+)\s*:?\s*$',
            line.strip(),
        )
        if not match:
            raise self._error(
                "Expected: show card \"Title\" from dataset|table|frame SOURCE",
                self.pos,
                line,
                hint='Card components require a title and data source, e.g., show card "Appointments" from dataset appointments'
            )
        title = match.group(1)
        source_type = match.group(2)
        source = match.group(3)
        
        empty_state: Optional[EmptyStateConfig] = None
        item_config: Optional[CardItemConfig] = None
        group_by: Optional[str] = None
        filter_by: Optional[str] = None
        sort_by: Optional[str] = None
        layout: Optional[str] = None
        binding: Optional[str] = None
        variant = None
        tone = None
        density = None
        size = None
        
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
                
            if stripped.startswith('empty_state:'):
                self._advance()
                empty_state = self._parse_empty_state(indent)
            elif stripped.startswith('item:'):
                self._advance()
                item_config = self._parse_card_item_config(indent)
            elif stripped.startswith('group_by:'):
                group_by = stripped[len('group_by:'):].strip()
                self._advance()
            elif stripped.startswith('filter_by:') or stripped.startswith('filter by:') or stripped.startswith('condition:'):
                # Support both filter_by: and condition: as aliases
                if stripped.startswith('filter_by:'):
                    key_len = len('filter_by:')
                elif stripped.startswith('filter by:'):
                    key_len = len('filter by:')
                else:
                    key_len = len('condition:')
                filter_by = stripped[key_len:].strip()
                self._advance()
            elif stripped.startswith('sort_by:') or stripped.startswith('sort by:'):
                key_len = len('sort_by:') if stripped.startswith('sort_by:') else len('sort by:')
                sort_by = stripped[key_len:].strip()
                self._advance()
            elif stripped.startswith('layout:'):
                layout = stripped[len('layout:'):].strip()
                self._advance()
            elif stripped.startswith('binding:'):
                binding = stripped[len('binding:'):].strip()
                self._advance()
            elif stripped.startswith('variant:'):
                value = stripped[len('variant:'):].strip()
                variant = self._parse_design_token(value, "variant", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('tone:'):
                value = stripped[len('tone:'):].strip()
                tone = self._parse_design_token(value, "tone", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('density:'):
                value = stripped[len('density:'):].strip()
                density = self._parse_design_token(value, "density", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('size:'):
                value = stripped[len('size:'):].strip()
                size = self._parse_design_token(value, "size", self.pos + 1, nxt)
                self._advance()
            else:
                raise self._error(
                    "Expected card property (empty_state, item, group_by, filter_by, sort_by, layout, binding, variant, tone, density, or size)",
                    self.pos + 1,
                    nxt,
                    hint='Valid card properties: empty_state, item, group_by, filter_by, sort_by, layout, binding, variant, tone, density, size'
                )
        
        return ShowCard(
            title=title,
            source_type=source_type,
            source=source,
            empty_state=empty_state,
            item_config=item_config,
            group_by=group_by,
            filter_by=filter_by,
            sort_by=sort_by,
            layout=layout,
            binding=binding,
            variant=variant,
            tone=tone,
            density=density,
            size=size,
        )

    def _parse_show_list(self, line: str, base_indent: int) -> ShowList:
        """
        Parse a list display component.
        
        Syntax: show list "Title" from dataset|table|frame SOURCE
        
        Similar to show card but with list-specific features like search and filters.
        """
        match = re.match(
            r'show\s+list\s+"([^\"]+)"\s+from\s+(dataset|table|frame)\s+([^\s:]+)\s*:?\s*$',
            line.strip(),
        )
        if not match:
            raise self._error(
                "Expected: show list \"Title\" from dataset|table|frame SOURCE",
                self.pos,
                line,
                hint='List components require a title and data source, e.g., show list "Messages" from dataset messages'
            )
        title = match.group(1)
        source_type = match.group(2)
        source = match.group(3)
        
        list_type: Optional[str] = None
        empty_state: Optional[EmptyStateConfig] = None
        item_config: Optional[CardItemConfig] = None
        enable_search: bool = False
        filters: Optional[List[Dict[str, Any]]] = None
        columns: Optional[int] = None
        group_by: Optional[str] = None
        filter_by: Optional[str] = None
        sort_by: Optional[str] = None
        variant = None
        tone = None
        density = None
        size = None
        
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
                
            if stripped.startswith('list_type:'):
                list_type = stripped[len('list_type:'):].strip()
                self._advance()
            elif stripped.startswith('empty_state:'):
                self._advance()
                empty_state = self._parse_empty_state(indent)
            elif stripped.startswith('item:'):
                self._advance()
                item_config = self._parse_card_item_config(indent)
            elif stripped.startswith('enable_search:'):
                val = stripped[len('enable_search:'):].strip().lower()
                enable_search = val in ('true', 'yes', '1')
                self._advance()
            elif stripped.startswith('columns:'):
                try:
                    columns = int(stripped[len('columns:'):].strip())
                except ValueError:
                    raise self._error(
                        "columns must be an integer",
                        self.pos + 1,
                        nxt
                    )
                self._advance()
            elif stripped.startswith('group_by:'):
                group_by = stripped[len('group_by:'):].strip()
                self._advance()
            elif stripped.startswith('filter_by:') or stripped.startswith('filter by:'):
                key_len = len('filter_by:') if stripped.startswith('filter_by:') else len('filter by:')
                filter_by = stripped[key_len:].strip()
                self._advance()
            elif stripped.startswith('sort_by:') or stripped.startswith('sort by:'):
                key_len = len('sort_by:') if stripped.startswith('sort_by:') else len('sort by:')
                sort_by = stripped[key_len:].strip()
                self._advance()
            elif stripped.startswith('variant:'):
                value = stripped[len('variant:'):].strip()
                variant = self._parse_design_token(value, "variant", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('tone:'):
                value = stripped[len('tone:'):].strip()
                tone = self._parse_design_token(value, "tone", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('density:'):
                value = stripped[len('density:'):].strip()
                density = self._parse_design_token(value, "density", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('size:'):
                value = stripped[len('size:'):].strip()
                size = self._parse_design_token(value, "size", self.pos + 1, nxt)
                self._advance()
            else:
                raise self._error(
                    "Expected list property (list_type, empty_state, item, enable_search, columns, group_by, filter_by, sort_by, variant, tone, density, or size)",
                    self.pos + 1,
                    nxt,
                    hint='Valid list properties: list_type, empty_state, item, enable_search, columns, group_by, filter_by, sort_by, variant, tone, density, size'
                )
        
        return ShowList(
            title=title,
            source_type=source_type,
            source=source,
            list_type=list_type,
            empty_state=empty_state,
            item_config=item_config,
            enable_search=enable_search,
            filters=filters,
            columns=columns,
            group_by=group_by,
            filter_by=filter_by,
            sort_by=sort_by,
            variant=variant,
            tone=tone,
            density=density,
            size=size,
        )

    # =============================================================================
    # Data Display Component Parsers
    # =============================================================================

    def _parse_show_data_table(self, line: str, base_indent: int) -> ShowDataTable:
        """
        Parse a data_table display component.
        
        Syntax: show data_table "Title" from dataset|table|frame SOURCE
        
        Supports columns with configuration, row_actions, toolbar with search/filters/bulk_actions.
        """
        match = re.match(
            r'show\s+data_table\s+"([^\"]+)"\s+from\s+(dataset|table|frame)\s+([^\s:]+)\s*:?\s*$',
            line.strip(),
        )
        if not match:
            raise self._error(
                "Expected: show data_table \"Title\" from dataset|table|frame SOURCE",
                self.pos,
                line,
                hint='Data table components require a title and data source'
            )
        
        title = match.group(1)
        source_type = match.group(2)
        source = match.group(3)
        
        columns: List[ColumnConfig] = []
        row_actions: List[ConditionalAction] = []
        toolbar: Optional[ToolbarConfig] = None
        filter_by: Optional[str] = None
        sort_by: Optional[str] = None
        default_sort: Optional[Dict[str, str]] = None
        page_size: int = 50
        enable_pagination: bool = True
        empty_state: Optional[EmptyStateConfig] = None
        layout_meta: Optional[LayoutMeta] = None
        style: Optional[Dict[str, Any]] = None
        
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
            
            if stripped.startswith('columns:'):
                self._advance()
                columns = self._parse_table_columns(indent)
            elif stripped.startswith('row_actions:'):
                self._advance()
                row_actions = self._parse_conditional_actions(indent)
            elif stripped.startswith('toolbar:'):
                self._advance()
                toolbar = self._parse_toolbar_config(indent)
            elif stripped.startswith('filter_by:') or stripped.startswith('filter by:'):
                key_len = len('filter_by:') if stripped.startswith('filter_by:') else len('filter by:')
                filter_by = stripped[key_len:].strip()
                self._advance()
            elif stripped.startswith('sort_by:') or stripped.startswith('sort by:'):
                key_len = len('sort_by:') if stripped.startswith('sort_by:') else len('sort by:')
                sort_by = stripped[key_len:].strip()
                self._advance()
            elif stripped.startswith('default_sort:'):
                self._advance()
                default_sort = self._parse_default_sort(indent)
            elif stripped.startswith('page_size:'):
                try:
                    page_size = int(stripped[len('page_size:'):].strip())
                except ValueError:
                    raise self._error("page_size must be an integer", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('enable_pagination:'):
                val = stripped[len('enable_pagination:'):].strip().lower()
                enable_pagination = val in ('true', 'yes', '1')
                self._advance()
            elif stripped.startswith('empty_state:'):
                self._advance()
                empty_state = self._parse_empty_state(indent)
            elif stripped.startswith('layout:'):
                self._advance()
                block = self._parse_kv_block(indent)
                layout_meta = self._build_layout_meta(block)
            elif stripped.startswith('style:'):
                self._advance()
                style = self._parse_kv_block(indent)
            else:
                raise self._error(
                    "Expected data_table property (columns, row_actions, toolbar, filter_by, etc.)",
                    self.pos + 1,
                    nxt,
                    hint='Valid properties: columns, row_actions, toolbar, filter_by, sort_by, page_size, empty_state'
                )
        
        return ShowDataTable(
            title=title,
            source_type=source_type,
            source=source,
            columns=columns,
            row_actions=row_actions,
            toolbar=toolbar,
            filter_by=filter_by,
            sort_by=sort_by,
            default_sort=default_sort,
            page_size=page_size,
            enable_pagination=enable_pagination,
            empty_state=empty_state,
            layout=layout_meta,
            style=style,
        )

    def _parse_show_data_list(self, line: str, base_indent: int) -> ShowDataList:
        """
        Parse a data_list display component.
        
        Syntax: show data_list "Title" from dataset|table|frame SOURCE
        
        Supports item configuration with avatar, title, subtitle, metadata, actions.
        """
        match = re.match(
            r'show\s+data_list\s+"([^\"]+)"\s+from\s+(dataset|table|frame)\s+([^\s:]+)\s*:?\s*$',
            line.strip(),
        )
        if not match:
            raise self._error(
                "Expected: show data_list \"Title\" from dataset|table|frame SOURCE",
                self.pos,
                line,
                hint='Data list components require a title and data source'
            )
        
        title = match.group(1)
        source_type = match.group(2)
        source = match.group(3)
        
        item: Optional[ListItemConfig] = None
        variant: str = "default"
        dividers: bool = True
        filter_by: Optional[str] = None
        enable_search: bool = False
        search_placeholder: Optional[str] = None
        page_size: int = 50
        enable_pagination: bool = True
        empty_state: Optional[EmptyStateConfig] = None
        layout_meta: Optional[LayoutMeta] = None
        style: Optional[Dict[str, Any]] = None
        
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
            
            if stripped.startswith('item:') or stripped.startswith('item_config:'):
                self._advance()
                item = self._parse_list_item_config(indent)
            elif stripped.startswith('row_actions:'):
                # Support explicit row action definitions
                self._advance()
                actions = self._parse_conditional_actions(indent)
                if item is None:
                    item = ListItemConfig(title="", actions=actions)
                else:
                    item.actions = actions
            elif stripped.startswith('variant:'):
                variant = stripped[len('variant:'):].strip()
                self._advance()
            elif stripped.startswith('dividers:'):
                val = stripped[len('dividers:'):].strip().lower()
                dividers = val in ('true', 'yes', '1')
                self._advance()
            elif stripped.startswith('filter_by:'):
                filter_by = stripped[len('filter_by:'):].strip()
                self._advance()
            elif stripped.startswith('enable_search:'):
                val = stripped[len('enable_search:'):].strip().lower()
                enable_search = val in ('true', 'yes', '1')
                self._advance()
            elif stripped.startswith('search_placeholder:'):
                search_placeholder = stripped[len('search_placeholder:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('page_size:'):
                try:
                    page_size = int(stripped[len('page_size:'):].strip())
                except ValueError:
                    raise self._error("page_size must be an integer", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('enable_pagination:'):
                val = stripped[len('enable_pagination:'):].strip().lower()
                enable_pagination = val in ('true', 'yes', '1')
                self._advance()
            elif stripped.startswith('empty_state:'):
                self._advance()
                empty_state = self._parse_empty_state(indent)
            elif stripped.startswith('layout:'):
                self._advance()
                block = self._parse_kv_block(indent)
                layout_meta = self._build_layout_meta(block)
            elif stripped.startswith('style:'):
                self._advance()
                style = self._parse_kv_block(indent)
            else:
                raise self._error(
                    "Expected data_list property (item, variant, dividers, filter_by, etc.)",
                    self.pos + 1,
                    nxt
                )
        
        return ShowDataList(
            title=title,
            source_type=source_type,
            source=source,
            item=item,
            variant=variant,
            dividers=dividers,
            filter_by=filter_by,
            enable_search=enable_search,
            search_placeholder=search_placeholder,
            page_size=page_size,
            enable_pagination=enable_pagination,
            empty_state=empty_state,
            layout=layout_meta,
            style=style,
        )

    def _parse_show_stat_summary(self, line: str, base_indent: int) -> ShowStatSummary:
        """
        Parse a stat_summary display component.
        
        Syntax: show stat_summary "Label" from dataset|table|frame SOURCE
        
        Supports value, format, delta, trend, sparkline configuration.
        """
        match = re.match(
            r'show\s+stat_summary\s+"([^\"]+)"\s+from\s+(dataset|table|frame)\s+([^\s:]+)\s*:?\s*$',
            line.strip(),
        )
        if not match:
            raise self._error(
                "Expected: show stat_summary \"Label\" from dataset|table|frame SOURCE",
                self.pos,
                line,
                hint='Stat summary components require a label and data source'
            )
        
        label = match.group(1)
        source_type = match.group(2)
        source = match.group(3)
        
        value: str = ""
        format_str: Optional[str] = None
        prefix: Optional[str] = None
        suffix: Optional[str] = None
        delta: Optional[Dict[str, Any]] = None
        trend: Optional[Union[str, Dict[str, Any]]] = None
        comparison_period: Optional[str] = None
        sparkline: Optional[SparklineConfig] = None
        color: Optional[str] = None
        icon: Optional[str] = None
        layout_meta: Optional[LayoutMeta] = None
        style: Optional[Dict[str, Any]] = None
        stats: List[Dict[str, Any]] = []
        
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
            
            if stripped.startswith('stats:'):
                self._advance()
                stats = self._parse_stats_list(indent)
                if stats:
                    primary = stats[0]
                    value = primary.get('value', value)
                    format_str = primary.get('format', format_str)
                    prefix = primary.get('prefix', prefix)
                    suffix = primary.get('suffix', suffix)
                    delta = primary.get('delta', delta)
                    trend = primary.get('trend', trend)
                    comparison_period = primary.get('comparison_period', comparison_period)
                    color = primary.get('color', color)
                    icon = primary.get('icon', icon)
            elif stripped.startswith('value:'):
                value = stripped[len('value:'):].strip()
                self._advance()
            elif stripped.startswith('format:'):
                format_str = stripped[len('format:'):].strip()
                self._advance()
            elif stripped.startswith('prefix:'):
                prefix = stripped[len('prefix:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('suffix:'):
                suffix = stripped[len('suffix:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('delta:'):
                self._advance()
                delta = self._parse_kv_block(indent)
            elif stripped.startswith('trend:'):
                trend_val = stripped[len('trend:'):].strip()
                if trend_val:
                    trend = trend_val
                else:
                    self._advance()
                    trend = self._parse_kv_block(indent)
                self._advance() if trend_val else None
            elif stripped.startswith('comparison_period:'):
                comparison_period = stripped[len('comparison_period:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('sparkline:'):
                self._advance()
                sparkline = self._parse_sparkline_config(indent)
            elif stripped.startswith('color:'):
                color = stripped[len('color:'):].strip()
                self._advance()
            elif stripped.startswith('icon:'):
                icon = stripped[len('icon:'):].strip()
                self._advance()
            elif stripped.startswith('layout:'):
                self._advance()
                block = self._parse_kv_block(indent)
                layout_meta = self._build_layout_meta(block)
            elif stripped.startswith('style:'):
                self._advance()
                style = self._parse_kv_block(indent)
            else:
                raise self._error(
                    "Expected stat_summary property (value, format, delta, trend, sparkline, etc.)",
                    self.pos + 1,
                    nxt
                )
        
        summary = ShowStatSummary(
            label=label,
            source_type=source_type,
            source=source,
            value=value,
            format=format_str,
            prefix=prefix,
            suffix=suffix,
            delta=delta,
            trend=trend,
            comparison_period=comparison_period,
            sparkline=sparkline,
            color=color,
            icon=icon,
            layout=layout_meta,
            style=style,
            stats=stats or None,
        )

        summary.title = label
        return summary

    def _parse_stats_list(self, base_indent: int) -> List[Dict[str, Any]]:
        """Parse a list of stat configurations from a stats: block."""
        stats: List[Dict[str, Any]] = []

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

            if stripped.startswith('-'):
                self._advance()
                entry_indent = indent
                entry: Dict[str, Any] = {}

                inline = stripped[1:].strip()
                if inline and ':' in inline:
                    key, val = inline.split(':', 1)
                    entry[key.strip()] = val.strip().strip('"')

                entry.update(self._parse_kv_block(entry_indent))
                stats.append(entry)
            else:
                # If the line isn't a list item, stop processing stats
                break

        return stats

    def _parse_show_timeline(self, line: str, base_indent: int) -> ShowTimeline:
        """
        Parse a timeline display component.
        
        Syntax: show timeline "Title" from dataset|table|frame SOURCE
        
        Supports item configuration with timestamp, title, description, icon, status, actions.
        """
        match = re.match(
            r'show\s+timeline\s+"([^\"]+)"\s+from\s+(dataset|table|frame)\s+([^\s:]+)\s*:?\s*$',
            line.strip(),
        )
        if not match:
            raise self._error(
                "Expected: show timeline \"Title\" from dataset|table|frame SOURCE",
                self.pos,
                line,
                hint='Timeline components require a title and data source'
            )
        
        title = match.group(1)
        source_type = match.group(2)
        source = match.group(3)
        
        item: Optional[TimelineItem] = None
        variant: str = "default"
        show_timestamps: bool = True
        group_by_date: bool = False
        filter_by: Optional[str] = None
        sort_by: Optional[str] = None
        page_size: int = 50
        enable_pagination: bool = True
        empty_state: Optional[EmptyStateConfig] = None
        layout_meta: Optional[LayoutMeta] = None
        style: Optional[Dict[str, Any]] = None
        
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
            
            if stripped.startswith('item:') or stripped.startswith('item_config:'):
                self._advance()
                item = self._parse_timeline_item(indent)
            elif stripped.startswith('variant:'):
                variant = stripped[len('variant:'):].strip()
                self._advance()
            elif stripped.startswith('show_timestamps:'):
                val = stripped[len('show_timestamps:'):].strip().lower()
                show_timestamps = val in ('true', 'yes', '1')
                self._advance()
            elif stripped.startswith('group_by_date:'):
                val = stripped[len('group_by_date:'):].strip().lower()
                group_by_date = val in ('true', 'yes', '1')
                self._advance()
            elif stripped.startswith('filter_by:'):
                filter_by = stripped[len('filter_by:'):].strip()
                self._advance()
            elif stripped.startswith('sort_by:'):
                sort_by = stripped[len('sort_by:'):].strip()
                self._advance()
            elif stripped.startswith('page_size:'):
                try:
                    page_size = int(stripped[len('page_size:'):].strip())
                except ValueError:
                    raise self._error("page_size must be an integer", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('enable_pagination:'):
                val = stripped[len('enable_pagination:'):].strip().lower()
                enable_pagination = val in ('true', 'yes', '1')
                self._advance()
            elif stripped.startswith('empty_state:'):
                self._advance()
                empty_state = self._parse_empty_state(indent)
            elif stripped.startswith('layout:'):
                self._advance()
                block = self._parse_kv_block(indent)
                layout_meta = self._build_layout_meta(block)
            elif stripped.startswith('style:'):
                self._advance()
                style = self._parse_kv_block(indent)
            else:
                raise self._error(
                    "Expected timeline property (item, variant, show_timestamps, etc.)",
                    self.pos + 1,
                    nxt
                )
        
        return ShowTimeline(
            title=title,
            source_type=source_type,
            source=source,
            item=item,
            variant=variant,
            show_timestamps=show_timestamps,
            group_by_date=group_by_date,
            filter_by=filter_by,
            sort_by=sort_by,
            page_size=page_size,
            enable_pagination=enable_pagination,
            empty_state=empty_state,
            layout=layout_meta,
            style=style,
        )

    def _parse_show_avatar_group(self, line: str, base_indent: int) -> ShowAvatarGroup:
        """
        Parse an avatar_group display component.
        
        Syntax: show avatar_group ["Title"] from dataset|table|frame SOURCE
        
        Supports item configuration with name, image_url, initials, color, status.
        """
        match = re.match(
            r'show\s+avatar_group(?:\s+"([^\"]+)")?\s+from\s+(dataset|table|frame)\s+([^\s:]+)\s*:?\s*$',
            line.strip(),
        )
        if not match:
            raise self._error(
                "Expected: show avatar_group [\"Title\"] from dataset|table|frame SOURCE",
                self.pos,
                line,
                hint='Avatar group components require a data source'
            )
        
        title = match.group(1)  # Optional
        source_type = match.group(2)
        source = match.group(3)
        
        item: Optional[AvatarItem] = None
        max_visible: int = 5
        size: str = "md"
        variant: str = "stacked"
        filter_by: Optional[str] = None
        layout_meta: Optional[LayoutMeta] = None
        style: Optional[Dict[str, Any]] = None
        
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
            
            if stripped.startswith('item:'):
                self._advance()
                item = self._parse_avatar_item(indent)
            elif stripped.startswith('avatar_field:'):
                if item is None:
                    item = AvatarItem()
                item.image_url = stripped[len('avatar_field:'):].strip()
                self._advance()
            elif stripped.startswith('name_field:'):
                if item is None:
                    item = AvatarItem()
                item.name = stripped[len('name_field:'):].strip()
                self._advance()
            elif stripped.startswith('max_visible:'):
                try:
                    max_visible = int(stripped[len('max_visible:'):].strip())
                except ValueError:
                    raise self._error("max_visible must be an integer", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('max_display:'):
                try:
                    max_visible = int(stripped[len('max_display:'):].strip())
                except ValueError:
                    raise self._error("max_display must be an integer", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('size:'):
                size = stripped[len('size:'):].strip()
                self._advance()
            elif stripped.startswith('variant:'):
                variant = stripped[len('variant:'):].strip()
                self._advance()
            elif stripped.startswith('filter_by:'):
                filter_by = stripped[len('filter_by:'):].strip()
                self._advance()
            elif stripped.startswith('layout:'):
                self._advance()
                block = self._parse_kv_block(indent)
                layout_meta = self._build_layout_meta(block)
            elif stripped.startswith('style:'):
                self._advance()
                style = self._parse_kv_block(indent)
            else:
                raise self._error(
                    "Expected avatar_group property (item, max_visible, size, variant, etc.)",
                    self.pos + 1,
                    nxt
                )
        
        return ShowAvatarGroup(
            title=title,
            source_type=source_type,
            source=source,
            item=item,
            max_visible=max_visible,
            size=size,
            variant=variant,
            filter_by=filter_by,
            layout=layout_meta,
            style=style,
        )

    def _parse_show_data_chart(self, line: str, base_indent: int) -> ShowDataChart:
        """
        Parse a data_chart display component.
        
        Syntax: show data_chart "Title" from dataset|table|frame SOURCE
        
        Supports enhanced chart configuration with multi-series, variants, legend, tooltip, axes.
        """
        match = re.match(
            r'show\s+data_chart\s+"([^\"]+)"\s+from\s+(dataset|table|frame)\s+([^\s:]+)\s*:?\s*$',
            line.strip(),
        )
        if not match:
            raise self._error(
                "Expected: show data_chart \"Title\" from dataset|table|frame SOURCE",
                self.pos,
                line,
                hint='Data chart components require a title and data source'
            )
        
        title = match.group(1)
        source_type = match.group(2)
        source = match.group(3)
        
        config: Optional[ChartConfig] = None
        filter_by: Optional[str] = None
        sort_by: Optional[str] = None
        empty_state: Optional[EmptyStateConfig] = None
        layout_meta: Optional[LayoutMeta] = None
        style: Optional[Dict[str, Any]] = None
        height: Optional[Union[str, int]] = None
        
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
            
            if stripped.startswith('config:'):
                self._advance()
                config = self._parse_chart_config(indent)
            elif stripped.startswith('chart_type:'):
                if config is None:
                    config = ChartConfig(x_field="")
                config.variant = stripped[len('chart_type:'):].strip()
                self._advance()
            elif stripped.startswith('x_axis:'):
                if config is None:
                    config = ChartConfig(x_field="")
                config.x_field = stripped[len('x_axis:'):].strip()
                self._advance()
            elif stripped.startswith('y_axis:'):
                if config is None:
                    config = ChartConfig(x_field="")
                y_field = stripped[len('y_axis:'):].strip()
                if y_field:
                    config.y_fields.append(y_field)
                self._advance()
            elif stripped.startswith('group_by:'):
                if config is None:
                    config = ChartConfig(x_field="")
                config.group_by = stripped[len('group_by:'):].strip()
                self._advance()
            elif stripped.startswith('filter_by:'):
                filter_by = stripped[len('filter_by:'):].strip()
                self._advance()
            elif stripped.startswith('sort_by:'):
                sort_by = stripped[len('sort_by:'):].strip()
                self._advance()
            elif stripped.startswith('height:'):
                height_val = stripped[len('height:'):].strip()
                try:
                    height = int(height_val)
                except ValueError:
                    height = height_val
                self._advance()
            elif stripped.startswith('empty_state:'):
                self._advance()
                empty_state = self._parse_empty_state(indent)
            elif stripped.startswith('layout:'):
                self._advance()
                block = self._parse_kv_block(indent)
                layout_meta = self._build_layout_meta(block)
            elif stripped.startswith('style:'):
                self._advance()
                style = self._parse_kv_block(indent)
            else:
                raise self._error(
                    "Expected data_chart property (config, filter_by, sort_by, height, empty_state, etc.)",
                    self.pos + 1,
                    nxt
                )
        
        return ShowDataChart(
            title=title,
            source_type=source_type,
            source=source,
            config=config,
            filter_by=filter_by,
            sort_by=sort_by,
            empty_state=empty_state,
            layout=layout_meta,
            style=style,
            height=height,
        )

    # =============================================================================
    # Helper Parsers for Data Display Components
    # =============================================================================

    def _parse_table_columns(self, base_indent: int) -> List[ColumnConfig]:
        """Parse list of table column configurations."""
        columns: List[ColumnConfig] = []
        
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
            
            # Accept both "- id:" and "- field:" as column start
            if stripped.startswith('- id:') or stripped.startswith('-id:') or stripped.startswith('- field:') or stripped.startswith('-field:'):
                # Inline column definition
                col_id = ""
                initial_field = None
                
                if stripped.startswith('- field:') or stripped.startswith('-field:'):
                    # Extract field value as the initial field
                    initial_field = stripped.split(':', 1)[1].strip() if ':' in stripped else ""
                    col_id = initial_field  # Use field as ID initially
                else:
                    col_id = stripped.split(':', 1)[1].strip() if ':' in stripped else ""
                
                self._advance()
                
                # Parse column properties
                label: str = col_id
                field: Optional[str] = initial_field  # Start with field from dash line if present
                width: Optional[Union[str, int]] = None
                align: str = "left"
                sortable: bool = True
                format_str: Optional[str] = None
                transform: Optional[Union[str, Dict[str, Any]]] = None
                render_template: Optional[str] = None
                
                while self.pos < len(self.lines):
                    nxt2 = self._peek()
                    if nxt2 is None:
                        break
                    indent2 = self._indent(nxt2)
                    stripped2 = nxt2.strip()
                    if not stripped2 or stripped2.startswith('#'):
                        self._advance()
                        continue
                    if indent2 <= indent:
                        break
                    
                    if stripped2.startswith('label:') or stripped2.startswith('header:'):
                        key_len = len('header:') if stripped2.startswith('header:') else len('label:')
                        label = stripped2[key_len:].strip().strip('"')
                        self._advance()
                    elif stripped2.startswith('field:'):
                        field = stripped2[len('field:'):].strip()
                        self._advance()
                    elif stripped2.startswith('width:'):
                        width_val = stripped2[len('width:'):].strip()
                        try:
                            width = int(width_val)
                        except ValueError:
                            width = width_val
                        self._advance()
                    elif stripped2.startswith('align:'):
                        align = stripped2[len('align:'):].strip()
                        self._advance()
                    elif stripped2.startswith('sortable:'):
                        val = stripped2[len('sortable:'):].strip().lower()
                        sortable = val in ('true', 'yes', '1')
                        self._advance()
                    elif stripped2.startswith('format:'):
                        format_str = stripped2[len('format:'):].strip()
                        self._advance()
                    elif stripped2.startswith('transform:'):
                        transform_val = stripped2[len('transform:'):].strip()
                        if transform_val:
                            transform = transform_val
                        else:
                            self._advance()
                            transform = self._parse_kv_block(indent2)
                        self._advance() if transform_val else None
                    elif stripped2.startswith('render_template:'):
                        render_template = stripped2[len('render_template:'):].strip().strip('"')
                        self._advance()
                    else:
                        break
                
                columns.append(ColumnConfig(
                    id=col_id,
                    label=label,
                    field=field,
                    width=width,
                    align=align,
                    sortable=sortable,
                    format=format_str,
                    transform=transform,
                    render_template=render_template,
                ))
            else:
                break
        
        return columns

    def _parse_toolbar_config(self, base_indent: int) -> ToolbarConfig:
        """Parse toolbar configuration with search, filters, bulk_actions."""
        search: Optional[Dict[str, Any]] = None
        filters: List[Dict[str, Any]] = []
        bulk_actions: List[ConditionalAction] = []
        actions: List[ConditionalAction] = []
        
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
            
            if stripped.startswith('search:'):
                self._advance()
                search = self._parse_kv_block(indent)
            elif stripped.startswith('filters:'):
                self._advance()
                filters = self._parse_filters_list(indent)
            elif stripped.startswith('bulk_actions:'):
                self._advance()
                bulk_actions = self._parse_conditional_actions(indent)
            elif stripped.startswith('actions:'):
                self._advance()
                actions = self._parse_conditional_actions(indent)
            else:
                break
        
        return ToolbarConfig(
            search=search,
            filters=filters,
            bulk_actions=bulk_actions,
            actions=actions,
        )

    def _parse_filters_list(self, base_indent: int) -> List[Dict[str, Any]]:
        """Parse list of filter configurations."""
        filters: List[Dict[str, Any]] = []
        
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
            
            if stripped.startswith('- field:'):
                self._advance()
                filter_obj = self._parse_kv_block(indent)
                filters.append(filter_obj)
            else:
                break
        
        return filters

    def _parse_default_sort(self, base_indent: int) -> Dict[str, str]:
        """Parse default sort configuration."""
        sort_config: Dict[str, str] = {}
        
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
            
            if ':' in stripped:
                key, value = stripped.split(':', 1)
                sort_config[key.strip()] = value.strip()
                self._advance()
            else:
                break
        
        return sort_config

    def _parse_list_item_config(self, base_indent: int) -> ListItemConfig:
        """Parse list item configuration."""
        avatar: Optional[Dict[str, Any]] = None
        title: Union[str, Dict[str, Any]] = ""
        subtitle: Optional[Union[str, Dict[str, Any]]] = None
        metadata: Dict[str, Union[str, Dict[str, Any]]] = {}
        actions: List[ConditionalAction] = []
        badge: Optional[Union[BadgeConfig, Dict[str, Any]]] = None
        icon: Optional[str] = None
        state_class: Optional[Dict[str, str]] = None
        
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
            
            if stripped.startswith('avatar:'):
                self._advance()
                avatar = self._parse_kv_block(indent)
            elif stripped.startswith('title:'):
                title_val = stripped[len('title:'):].strip()
                if title_val:
                    title = title_val
                else:
                    self._advance()
                    title = self._parse_kv_block(indent)
                self._advance() if title_val else None
            elif stripped.startswith('subtitle:'):
                subtitle_val = stripped[len('subtitle:'):].strip()
                if subtitle_val:
                    subtitle = subtitle_val
                else:
                    self._advance()
                    subtitle = self._parse_kv_block(indent)
                self._advance() if subtitle_val else None
            elif stripped.startswith('metadata:'):
                self._advance()
                metadata = self._parse_kv_block(indent)
            elif stripped.startswith('actions:'):
                self._advance()
                actions = self._parse_conditional_actions(indent)
            elif stripped.startswith('badge:'):
                self._advance()
                badge = self._parse_kv_block(indent)
            elif stripped.startswith('icon:'):
                icon = stripped[len('icon:'):].strip()
                self._advance()
            elif stripped.startswith('state_class:'):
                self._advance()
                state_class = self._parse_kv_block(indent)
            else:
                break
        
        return ListItemConfig(
            avatar=avatar,
            title=title,
            subtitle=subtitle,
            metadata=metadata,
            actions=actions,
            badge=badge,
            icon=icon,
            state_class=state_class,
        )

    def _parse_sparkline_config(self, base_indent: int) -> SparklineConfig:
        """Parse sparkline configuration."""
        data_source: str = ""
        x_field: str = ""
        y_field: str = ""
        color: Optional[str] = None
        variant: str = "line"
        
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
            
            if stripped.startswith('data_source:'):
                data_source = stripped[len('data_source:'):].strip()
                self._advance()
            elif stripped.startswith('x_field:'):
                x_field = stripped[len('x_field:'):].strip()
                self._advance()
            elif stripped.startswith('y_field:'):
                y_field = stripped[len('y_field:'):].strip()
                self._advance()
            elif stripped.startswith('color:'):
                color = stripped[len('color:'):].strip()
                self._advance()
            elif stripped.startswith('variant:'):
                variant = stripped[len('variant:'):].strip()
                self._advance()
            else:
                break
        
        return SparklineConfig(
            data_source=data_source,
            x_field=x_field,
            y_field=y_field,
            color=color,
            variant=variant,
        )

    def _parse_timeline_item(self, base_indent: int) -> TimelineItem:
        """Parse timeline item configuration."""
        timestamp: Union[str, Dict[str, Any]] = ""
        title: Union[str, Dict[str, Any]] = ""
        description: Optional[Union[str, Dict[str, Any]]] = None
        icon: Optional[Union[str, Dict[str, Any]]] = None
        user: Optional[Union[str, Dict[str, Any]]] = None
        status: Optional[Union[str, Dict[str, Any]]] = None
        color: Optional[str] = None
        actions: List[ConditionalAction] = []
        
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
            
            if stripped.startswith('timestamp:'):
                timestamp_val = stripped[len('timestamp:'):].strip()
                if timestamp_val:
                    timestamp = timestamp_val
                else:
                    self._advance()
                    timestamp = self._parse_kv_block(indent)
                self._advance() if timestamp_val else None
            elif stripped.startswith('title:'):
                title_val = stripped[len('title:'):].strip()
                if title_val:
                    title = title_val
                else:
                    self._advance()
                    title = self._parse_kv_block(indent)
                self._advance() if title_val else None
            elif stripped.startswith('description:'):
                desc_val = stripped[len('description:'):].strip()
                if desc_val:
                    description = desc_val
                else:
                    self._advance()
                    description = self._parse_kv_block(indent)
                self._advance() if desc_val else None
            elif stripped.startswith('user:'):
                user_val = stripped[len('user:'):].strip()
                if user_val:
                    user = user_val
                else:
                    self._advance()
                    user = self._parse_kv_block(indent)
                self._advance() if user_val else None
            elif stripped.startswith('icon:'):
                icon_val = stripped[len('icon:'):].strip()
                if icon_val:
                    icon = icon_val
                else:
                    self._advance()
                    icon = self._parse_kv_block(indent)
                self._advance() if icon_val else None
            elif stripped.startswith('status:'):
                status_val = stripped[len('status:'):].strip()
                if status_val:
                    status = status_val
                else:
                    self._advance()
                    status = self._parse_kv_block(indent)
                self._advance() if status_val else None
            elif stripped.startswith('color:'):
                color = stripped[len('color:'):].strip()
                self._advance()
            elif stripped.startswith('actions:'):
                self._advance()
                actions = self._parse_conditional_actions(indent)
            else:
                break
        
        return TimelineItem(
            timestamp=timestamp,
            title=title,
            description=description,
            icon=icon,
            user=user,
            status=status,
            color=color,
            actions=actions,
        )

    def _parse_avatar_item(self, base_indent: int) -> AvatarItem:
        """Parse avatar item configuration."""
        name: Optional[Union[str, Dict[str, Any]]] = None
        image_url: Optional[Union[str, Dict[str, Any]]] = None
        initials: Optional[Union[str, Dict[str, Any]]] = None
        color: Optional[Union[str, Dict[str, Any]]] = None
        status: Optional[Union[str, Dict[str, Any]]] = None
        tooltip: Optional[Union[str, Dict[str, Any]]] = None
        
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
            
            if stripped.startswith('name:'):
                name_val = stripped[len('name:'):].strip()
                if name_val:
                    name = name_val
                else:
                    self._advance()
                    name = self._parse_kv_block(indent)
                self._advance() if name_val else None
            elif stripped.startswith('image_url:'):
                url_val = stripped[len('image_url:'):].strip()
                if url_val:
                    image_url = url_val
                else:
                    self._advance()
                    image_url = self._parse_kv_block(indent)
                self._advance() if url_val else None
            elif stripped.startswith('initials:'):
                initials_val = stripped[len('initials:'):].strip()
                if initials_val:
                    initials = initials_val
                else:
                    self._advance()
                    initials = self._parse_kv_block(indent)
                self._advance() if initials_val else None
            elif stripped.startswith('color:'):
                color_val = stripped[len('color:'):].strip()
                if color_val:
                    color = color_val
                else:
                    self._advance()
                    color = self._parse_kv_block(indent)
                self._advance() if color_val else None
            elif stripped.startswith('status:'):
                status_val = stripped[len('status:'):].strip()
                if status_val:
                    status = status_val
                else:
                    self._advance()
                    status = self._parse_kv_block(indent)
                self._advance() if status_val else None
            elif stripped.startswith('tooltip:'):
                tooltip_val = stripped[len('tooltip:'):].strip()
                if tooltip_val:
                    tooltip = tooltip_val
                else:
                    self._advance()
                    tooltip = self._parse_kv_block(indent)
                self._advance() if tooltip_val else None
            else:
                break
        
        return AvatarItem(
            name=name,
            image_url=image_url,
            initials=initials,
            color=color,
            status=status,
            tooltip=tooltip,
        )

    def _parse_chart_config(self, base_indent: int) -> ChartConfig:
        """Parse enhanced chart configuration."""
        variant: str = "line"
        x_field: str = ""
        y_fields: List[str] = []
        group_by: Optional[str] = None
        stacked: bool = False
        smooth: bool = True
        fill: bool = True
        legend: Optional[Dict[str, Any]] = None
        tooltip: Optional[Dict[str, Any]] = None
        x_axis: Optional[Dict[str, Any]] = None
        y_axis: Optional[Dict[str, Any]] = None
        colors: List[str] = []
        color_scheme: Optional[str] = None
        
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
            
            if stripped.startswith('variant:'):
                variant = stripped[len('variant:'):].strip()
                self._advance()
            elif stripped.startswith('x_field:'):
                x_field = stripped[len('x_field:'):].strip()
                self._advance()
            elif stripped.startswith('y_fields:'):
                fields_str = stripped[len('y_fields:'):].strip()
                if fields_str.startswith('[') and fields_str.endswith(']'):
                    fields_str = fields_str[1:-1]
                y_fields = [f.strip() for f in fields_str.split(',') if f.strip()]
                self._advance()
            elif stripped.startswith('group_by:'):
                group_by = stripped[len('group_by:'):].strip()
                self._advance()
            elif stripped.startswith('stacked:'):
                val = stripped[len('stacked:'):].strip().lower()
                stacked = val in ('true', 'yes', '1')
                self._advance()
            elif stripped.startswith('smooth:'):
                val = stripped[len('smooth:'):].strip().lower()
                smooth = val in ('true', 'yes', '1')
                self._advance()
            elif stripped.startswith('fill:'):
                val = stripped[len('fill:'):].strip().lower()
                fill = val in ('true', 'yes', '1')
                self._advance()
            elif stripped.startswith('legend:'):
                self._advance()
                legend = self._parse_kv_block(indent)
            elif stripped.startswith('tooltip:'):
                self._advance()
                tooltip = self._parse_kv_block(indent)
            elif stripped.startswith('x_axis:'):
                self._advance()
                x_axis = self._parse_kv_block(indent)
            elif stripped.startswith('y_axis:'):
                self._advance()
                y_axis = self._parse_kv_block(indent)
            elif stripped.startswith('colors:'):
                colors_str = stripped[len('colors:'):].strip()
                if colors_str.startswith('[') and colors_str.endswith(']'):
                    colors_str = colors_str[1:-1]
                colors = [c.strip() for c in colors_str.split(',') if c.strip()]
                self._advance()
            elif stripped.startswith('color_scheme:'):
                color_scheme = stripped[len('color_scheme:'):].strip()
                self._advance()
            else:
                break
        
        return ChartConfig(
            variant=variant,
            x_field=x_field,
            y_fields=y_fields,
            group_by=group_by,
            stacked=stacked,
            smooth=smooth,
            fill=fill,
            legend=legend,
            tooltip=tooltip,
            x_axis=x_axis,
            y_axis=y_axis,
            colors=colors,
            color_scheme=color_scheme,
        )

    def _parse_predict_statement(self, line: str, base_indent: int) -> PredictStatement:
        """
        Parse a prediction statement for ML model inference.
        
        Syntax: predict using model MODEL_NAME with dataset|table|payload|variables SOURCE into variable|dataset|insight TARGET
        
        Executes ML model predictions on input data and stores results.
        """
        stripped = line.strip()
        try:
            tokens = shlex.split(stripped)
        except ValueError as exc:
            raise self._error(
                f"Unable to parse predict statement: {exc}",
                self.pos,
                line,
                hint='Check for unmatched quotes in predict statement'
            )
        if not tokens or tokens[0] != 'predict':
            raise self._error(
                "Predict statements must start with 'predict'",
                self.pos,
                line,
                hint='Use: predict using model MODEL_NAME with dataset SOURCE'
            )

        idx = 1
        if idx < len(tokens) and tokens[idx].lower() == 'using':
            idx += 1
        if idx >= len(tokens) or tokens[idx].lower() != 'model':
            raise self._error(
                "Predict statements require 'using model'",
                self.pos,
                line,
                hint='Syntax: predict using model MODEL_NAME'
            )
        idx += 1
        if idx >= len(tokens):
            raise self._error(
                "Model name is required for predict statements",
                self.pos,
                line,
                hint='Specify the model to use, e.g., predict using model sales_forecast'
            )
        model_name = tokens[idx]
        idx += 1

        input_kind = 'dataset'
        input_ref: Optional[str] = None
        if idx < len(tokens) and tokens[idx].lower() == 'with':
            idx += 1
            if idx >= len(tokens):
                raise self._error("Expected input kind after 'with'", self.pos, line)
            possible_kind = tokens[idx].lower()
            if possible_kind in {'dataset', 'table', 'payload', 'variables'}:
                input_kind = possible_kind
                idx += 1
                if idx >= len(tokens):
                    raise self._error("Expected input reference after input kind", self.pos, line)
                input_ref = tokens[idx]
                idx += 1
            else:
                input_kind = 'dataset'
                input_ref = tokens[idx]
                idx += 1

        assign = InferenceTarget()
        if idx < len(tokens) and tokens[idx].lower() == 'into':
            idx += 1
            if idx >= len(tokens):
                raise self._error("Expected target after 'into'", self.pos, line)
            possible_target = tokens[idx].lower()
            if possible_target in {'variable', 'dataset', 'insight', 'component'}:
                target_kind = possible_target
                idx += 1
                if idx >= len(tokens):
                    raise self._error("Expected name after target kind", self.pos, line)
                target_name = tokens[idx]
                idx += 1
            else:
                target_kind = 'variable'
                target_name = tokens[idx]
                idx += 1
            assign = InferenceTarget(kind=target_kind, name=target_name)

        parameters: Dict[str, Any] = {}
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped_next = nxt.strip()
            if not stripped_next or stripped_next.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            lowered = stripped_next.lower()
            if lowered.startswith('params:') or lowered.startswith('parameters:') or lowered.startswith('options:'):
                block_indent = indent
                self._advance()
                block_data = self._parse_kv_block(block_indent)
                parameters.update(block_data)
            else:
                break

        return PredictStatement(
            model_name=model_name,
            input_kind=input_kind,
            input_ref=input_ref,
            assign=assign,
            parameters=parameters,
        )

    def _parse_variable_assignment(self, line: str, line_no: int, base_indent: int) -> VariableAssignment:
        """
        Parse a variable assignment statement.
        
        Syntax: set variable_name = expression
        
        Variables store computed values for later use in the page.
        """
        stripped = line.strip()
        if not stripped.startswith('set '):
            raise self._error(
                "Expected 'set' at start of variable assignment",
                line_no,
                line,
                hint='Variable assignments must start with "set", e.g., set total = price * quantity'
            )
        assignment = stripped[4:].strip()
        if '=' not in assignment:
            raise self._error(
                "Expected '=' in variable assignment: set name = expression",
                line_no,
                line,
                hint='Use assignment syntax: set variable_name = value_expression'
            )
        parts = assignment.split('=', 1)
        if len(parts) != 2:
            raise self._error(
                "Invalid variable assignment syntax",
                line_no,
                line,
                hint='Use: set variable_name = expression'
            )
        var_name = parts[0].strip()
        expr_text = parts[1].strip()
        if not var_name:
            raise self._error(
                "Variable name cannot be empty",
                line_no,
                line,
                hint='Provide a variable name before the = sign'
            )
        if not (var_name[0].isalpha() or var_name[0] == '_'):
            raise self._error(
                f"Variable name must start with letter or underscore: '{var_name}'",
                line_no,
                line,
                hint='Valid variable names: result, _temp, user_count (start with letter or underscore)'
            )
        for ch in var_name:
            if not (ch.isalnum() or ch == '_'):
                raise self._error(
                    f"Variable name can only contain letters, numbers, and underscores: '{var_name}'",
                    line_no,
                    line,
                    hint='Use only letters, numbers, and underscores in variable names'
                )
        if not expr_text:
            raise self._error(
                "Expression cannot be empty in variable assignment",
                line_no,
                line,
                hint='Provide a value or expression after the = sign'
            )
        try:
            expression = self._parse_expression(expr_text)
        except N3SyntaxError:
            raise
        except Exception as exc:
            raise self._error(f"Failed to parse expression: {exc}", line_no, line)
        return VariableAssignment(name=var_name, value=expression)

    def _parse_empty_state(self, base_indent: int) -> EmptyStateConfig:
        """
        Parse empty_state configuration block.
        
        Example:
            empty_state:
              icon: calendar
              icon_size: large
              title: "No appointments"
              message: "Your care team will schedule appointments as needed."
        """
        icon: Optional[str] = None
        icon_size: Optional[str] = None
        title: Optional[str] = None
        message: Optional[str] = None
        action_label: Optional[str] = None
        action_link: Optional[str] = None
        
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
                
            if ':' in stripped:
                key, value = stripped.split(':', 1)
                key = key.strip()
                value = value.strip().strip('"')
                
                if key == 'icon':
                    icon = value
                elif key == 'icon_size':
                    icon_size = value
                elif key == 'title':
                    title = value
                elif key == 'message':
                    message = value
                elif key == 'action_label':
                    action_label = value
                elif key == 'action_link':
                    action_link = value
                else:
                    raise self._error(
                        f"Unknown empty_state property: {key}",
                        self.pos + 1,
                        nxt,
                        hint='Valid properties: icon, icon_size, title, message, action_label, action_link'
                    )
                self._advance()
            else:
                break
        
        return EmptyStateConfig(
            icon=icon,
            icon_size=icon_size,
            title=title,
            message=message,
            action_label=action_label,
            action_link=action_link,
        )

    def _parse_card_item_config(self, base_indent: int) -> CardItemConfig:
        """
        Parse item configuration block.
        
        Example:
            item:
              type: card
              style: appointment_detail
              header:
                badges: [...]
              sections: [...]
              actions: [...]
              footer: {...}
        """
        item_type: Optional[str] = None
        style: Optional[str] = None
        state_class: Optional[Dict[str, str]] = None
        role_class: Optional[str] = None
        header: Optional[CardHeader] = None
        sections: Optional[List[CardSection]] = None
        actions: Optional[List[ConditionalAction]] = None
        footer: Optional[CardFooter] = None
        
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
                
            if stripped.startswith('type:'):
                item_type = stripped[len('type:'):].strip()
                self._advance()
            elif stripped.startswith('style:'):
                style = stripped[len('style:'):].strip()
                self._advance()
            elif stripped.startswith('state_class:'):
                self._advance()
                state_class = self._parse_kv_block(indent)
            elif stripped.startswith('role_class:'):
                role_class = stripped[len('role_class:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('header:'):
                self._advance()
                header = self._parse_card_header(indent)
            elif stripped.startswith('sections:'):
                self._advance()
                sections = self._parse_card_sections(indent)
            elif stripped.startswith('actions:'):
                self._advance()
                actions = self._parse_conditional_actions(indent)
            elif stripped.startswith('footer:'):
                self._advance()
                footer = self._parse_card_footer(indent)
            else:
                raise self._error(
                    f"Unknown item property: {stripped.split(':')[0]}",
                    self.pos + 1,
                    nxt,
                    hint='Valid properties: type, style, state_class, header, sections, actions, footer'
                )
        
        return CardItemConfig(
            type=item_type,
            style=style,
            state_class=state_class,
            role_class=role_class,
            header=header,
            sections=sections,
            actions=actions,
            footer=footer,
        )

    def _parse_card_header(self, base_indent: int) -> CardHeader:
        """Parse card header with title, subtitle, avatar, and badges."""
        title: Optional[str] = None
        subtitle: Optional[str] = None
        avatar: Optional[Dict[str, Any]] = None
        badges: Optional[List[BadgeConfig]] = None
        
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
                
            if stripped.startswith('title:'):
                title = stripped[len('title:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('subtitle:'):
                subtitle = stripped[len('subtitle:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('avatar:'):
                avatar_val = stripped[len('avatar:'):].strip()
                if avatar_val:
                    # Simple string value
                    avatar = avatar_val
                else:
                    # Nested block
                    self._advance()
                    avatar = self._parse_kv_block(indent)
                if not avatar_val:
                    pass  # already advanced
                else:
                    self._advance()
            elif stripped.startswith('badges:'):
                self._advance()
                badges = self._parse_badge_list(indent)
            else:
                break
        
        return CardHeader(
            title=title,
            subtitle=subtitle,
            avatar=avatar,
            badges=badges,
        )

    def _parse_badge_list(self, base_indent: int) -> List[BadgeConfig]:
        """Parse list of badge configurations."""
        badges = []
        
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
                
            if stripped.startswith('- '):
                # List item start
                badge = self._parse_badge_item(indent)
                badges.append(badge)
            else:
                break
        
        return badges

    def _parse_badge_item(self, base_indent: int) -> BadgeConfig:
        """Parse a single badge configuration."""
        field: Optional[str] = None
        text: Optional[str] = None
        style: Optional[str] = None
        transform: Optional[Any] = None
        icon: Optional[str] = None
        condition: Optional[str] = None
        variant = None
        tone = None
        size = None
        
        # First line starts with '-'
        line = self._peek()
        if line:
            stripped = line.strip()
            if stripped.startswith('- '):
                rest = stripped[2:].strip()
                if ':' in rest:
                    # Inline format: - field: status
                    key, value = rest.split(':', 1)
                    key = key.strip()
                    value = value.strip().strip('"')
                    if key == 'field':
                        field = value
                    elif key == 'text':
                        text = value
                self._advance()
        
        # Parse nested properties
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent or stripped.startswith('- '):
                break
                
            if ':' in stripped:
                key, value = stripped.split(':', 1)
                key = key.strip()
                value = value.strip().strip('"')
                
                if key == 'field':
                    field = value
                elif key == 'text':
                    text = value
                elif key == 'style':
                    style = value
                elif key == 'transform':
                    transform = value
                elif key == 'icon':
                    icon = value
                elif key == 'condition':
                    condition = value
                elif key == 'variant':
                    variant = self._parse_design_token(value, "variant", self.pos + 1, nxt)
                elif key == 'tone':
                    tone = self._parse_design_token(value, "tone", self.pos + 1, nxt)
                elif key == 'size':
                    size = self._parse_design_token(value, "size", self.pos + 1, nxt)
                self._advance()
            else:
                break
        
        return BadgeConfig(
            field=field,
            text=text,
            style=style,
            transform=transform,
            icon=icon,
            condition=condition,
            variant=variant,
            tone=tone,
            size=size,
        )

    def _parse_card_sections(self, base_indent: int) -> List[CardSection]:
        """Parse list of card sections."""
        sections = []
        
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
                
            if stripped.startswith('- '):
                section = self._parse_card_section(indent)
                sections.append(section)
            else:
                break
        
        return sections

    def _parse_card_section(self, base_indent: int) -> CardSection:
        """Parse a single card section."""
        section_type: Optional[str] = None
        condition: Optional[str] = None
        style: Optional[str] = None
        title: Optional[str] = None
        icon: Optional[str] = None
        columns: Optional[int] = None
        items: Optional[List[Any]] = None
        content: Optional[Dict[str, Any]] = None
        
        # First line starts with '-'
        line = self._peek()
        if line:
            stripped = line.strip()
            if stripped.startswith('- '):
                rest = stripped[2:].strip()
                if rest.startswith('type:'):
                    section_type = rest[len('type:'):].strip()
                self._advance()
        
        # Parse nested properties
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent or stripped.startswith('- '):
                break
                
            if stripped.startswith('type:'):
                section_type = stripped[len('type:'):].strip()
                self._advance()
            elif stripped.startswith('condition:'):
                condition = stripped[len('condition:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('style:'):
                style = stripped[len('style:'):].strip()
                self._advance()
            elif stripped.startswith('title:'):
                title = stripped[len('title:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('icon:'):
                icon = stripped[len('icon:'):].strip()
                self._advance()
            elif stripped.startswith('columns:'):
                try:
                    columns = int(stripped[len('columns:'):].strip())
                except ValueError:
                    pass
                self._advance()
            elif stripped.startswith('items:'):
                self._advance()
                items = self._parse_info_grid_items(indent)
            elif stripped.startswith('content:'):
                self._advance()
                content = self._parse_kv_block(indent)
            else:
                break
        
        return CardSection(
            type=section_type,
            condition=condition,
            style=style,
            title=title,
            icon=icon,
            columns=columns,
            items=items,
            content=content,
        )

    def _parse_info_grid_items(self, base_indent: int) -> List[InfoGridItem]:
        """Parse list of info grid items."""
        items = []
        
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
                
            if stripped.startswith('- '):
                item = self._parse_info_grid_item(indent)
                items.append(item)
            else:
                break
        
        return items

    def _parse_info_grid_item(self, base_indent: int) -> InfoGridItem:
        """Parse a single info grid item."""
        icon: Optional[str] = None
        label: Optional[str] = None
        field: Optional[str] = None
        values: Optional[List[FieldValueConfig]] = None
        
        # First line starts with '-'
        line = self._peek()
        if line:
            stripped = line.strip()
            if stripped.startswith('- '):
                rest = stripped[2:].strip()
                if rest.startswith('icon:'):
                    icon = rest[len('icon:'):].strip()
                self._advance()
        
        # Parse nested properties
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent or stripped.startswith('- '):
                break
                
            if stripped.startswith('icon:'):
                icon = stripped[len('icon:'):].strip()
                self._advance()
            elif stripped.startswith('label:'):
                label = stripped[len('label:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('field:'):
                field = stripped[len('field:'):].strip()
                self._advance()
            elif stripped.startswith('values:'):
                self._advance()
                values = self._parse_field_value_list(indent)
            else:
                break
        
        return InfoGridItem(
            icon=icon,
            label=label,
            field_name=field,
            values=values,
        )

    def _parse_field_value_list(self, base_indent: int) -> List[FieldValueConfig]:
        """Parse list of field value configurations."""
        values = []
        
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
                
            if stripped.startswith('- '):
                value = self._parse_field_value(indent)
                values.append(value)
            else:
                break
        
        return values

    def _parse_field_value(self, base_indent: int) -> FieldValueConfig:
        """Parse a single field value configuration."""
        field: Optional[str] = None
        text: Optional[str] = None
        format_str: Optional[str] = None
        style: Optional[str] = None
        transform: Optional[Any] = None
        
        # First line starts with '-'
        line = self._peek()
        if line:
            stripped = line.strip()
            if stripped.startswith('- '):
                rest = stripped[2:].strip()
                if rest.startswith('field:'):
                    field = rest[len('field:'):].strip()
                elif rest.startswith('text:'):
                    text = rest[len('text:'):].strip().strip('"')
                self._advance()
        
        # Parse nested properties
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent or stripped.startswith('- '):
                break
                
            if stripped.startswith('field:'):
                field = stripped[len('field:'):].strip()
                self._advance()
            elif stripped.startswith('text:'):
                text = stripped[len('text:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('format:'):
                format_str = stripped[len('format:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('style:'):
                style = stripped[len('style:'):].strip()
                self._advance()
            elif stripped.startswith('transform:'):
                transform = stripped[len('transform:'):].strip()
                self._advance()
            else:
                break
        
        return FieldValueConfig(
            field=field,
            text=text,
            format=format_str,
            style=style,
            transform=transform,
        )

    def _parse_conditional_actions(self, base_indent: int) -> List[ConditionalAction]:
        """Parse list of conditional actions."""
        actions = []
        
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
                
            if stripped.startswith('- '):
                action = self._parse_conditional_action(indent)
                actions.append(action)
            else:
                break
        
        return actions

    def _parse_conditional_action(self, base_indent: int) -> ConditionalAction:
        """Parse a single conditional action."""
        label: Optional[str] = None
        icon: Optional[str] = None
        style: Optional[str] = None
        action: Optional[str] = None
        link: Optional[str] = None
        params: Optional[str] = None
        condition: Optional[str] = None
        variant = None
        tone = None
        size = None
        
        # First line starts with '-'
        line = self._peek()
        if line:
            stripped = line.strip()
            if stripped.startswith('- '):
                rest = stripped[2:].strip()
                if rest.startswith('label:'):
                    label = rest[len('label:'):].strip().strip('"')
                self._advance()
        
        # Parse nested properties
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent or stripped.startswith('- '):
                break
                
            if stripped.startswith('label:'):
                label = stripped[len('label:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('icon:'):
                icon = stripped[len('icon:'):].strip()
                self._advance()
            elif stripped.startswith('style:'):
                style = stripped[len('style:'):].strip()
                self._advance()
            elif stripped.startswith('action:'):
                action = stripped[len('action:'):].strip()
                self._advance()
            elif stripped.startswith('link:'):
                link = stripped[len('link:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('params:'):
                params = stripped[len('params:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('condition:'):
                condition = stripped[len('condition:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('variant:'):
                value = stripped[len('variant:'):].strip()
                variant = self._parse_design_token(value, "variant", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('tone:'):
                value = stripped[len('tone:'):].strip()
                tone = self._parse_design_token(value, "tone", self.pos + 1, nxt)
                self._advance()
            elif stripped.startswith('size:'):
                value = stripped[len('size:'):].strip()
                size = self._parse_design_token(value, "size", self.pos + 1, nxt)
                self._advance()
            else:
                break
        
        return ConditionalAction(
            label=label,
            icon=icon,
            style=style,
            action=action,
            link=link,
            params=params,
            condition=condition,
            variant=variant,
            tone=tone,
            size=size,
        )

    def _parse_card_footer(self, base_indent: int) -> CardFooter:
        """Parse card footer configuration."""
        text: Optional[str] = None
        condition: Optional[str] = None
        style: Optional[str] = None
        left: Optional[Dict[str, str]] = None
        right: Optional[Dict[str, str]] = None
        
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
                
            if stripped.startswith('text:'):
                text = stripped[len('text:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('condition:'):
                condition = stripped[len('condition:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('style:'):
                style = stripped[len('style:'):].strip()
                self._advance()
            elif stripped.startswith('left:'):
                self._advance()
                left = self._parse_kv_block(indent)
            elif stripped.startswith('right:'):
                self._advance()
                right = self._parse_kv_block(indent)
            else:
                break
        
        return CardFooter(
            text=text,
            condition=condition,
            style=style,
            left=left,
            right=right,
        )

    # =========================================================================
    # Layout Primitives (Stack, Grid, Split, Tabs, Accordion)
    # =========================================================================
    
    def _parse_children_list(self, base_indent: int) -> List[PageStatement]:
        """Parse a list of child page statements, handling dash-list syntax.
        
        Supports both:
            children:
              - show card "A"
              - show card "B"
        and:
            children:
              show card "A"
              show card "B"
        """
        children: List[PageStatement] = []
        
        while self.pos < len(self.lines):
            child_line = self._peek()
            if child_line is None:
                break
            child_indent = self._indent(child_line)
            child_stripped = child_line.strip()
            
            if child_indent <= base_indent or not child_stripped:
                if not child_stripped:
                    self._advance()
                    continue
                break
            
            # Handle dash-list items: "- show ..." or "- layout ..."
            if child_stripped.startswith('- '):
                # Remove the dash and parse the statement
                statement_text = child_stripped[2:].strip()
                # Temporarily replace the line without the dash for parsing
                current_pos = self.pos
                original_line = self.lines[current_pos]
                self.lines[current_pos] = ' ' * child_indent + statement_text
                try:
                    child_stmt = self._parse_page_statement(child_indent)
                    if child_stmt:
                        children.append(child_stmt)
                finally:
                    # Restore original line (position may have advanced)
                    self.lines[current_pos] = original_line
            else:
                # Regular statement without dash
                child_stmt = self._parse_page_statement(child_indent)
                if child_stmt:
                    children.append(child_stmt)
        
        return children

    def _parse_layout_stack(self, line: str, base_indent: int) -> StackLayout:
        """
        Parse a stack layout component.
        
        Syntax: layout stack:
                    direction: vertical
                    gap: medium
                    align: center
                    justify: start
                    wrap: false
                    children:
                        - show card "Data" from dataset items
                        - show chart "Chart" from dataset metrics
        """
        direction = "vertical"
        gap: Union[str, int] = "medium"
        align = "stretch"
        justify = "start"
        wrap = False
        children: List[Any] = []
        style: Optional[Dict[str, Any]] = None
        layout: Optional[LayoutMeta] = None
        
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
            
            if stripped.startswith('direction:'):
                direction = stripped[len('direction:'):].strip()
                if direction not in ('vertical', 'horizontal'):
                    raise self._error(
                        f"Invalid direction '{direction}'. Must be 'vertical' or 'horizontal'.",
                        self.pos + 1,
                        nxt
                    )
                self._advance()
            elif stripped.startswith('gap:'):
                gap_str = stripped[len('gap:'):].strip()
                # Try to parse as int, otherwise keep as string token
                try:
                    gap = int(gap_str)
                except ValueError:
                    gap = gap_str
                self._advance()
            elif stripped.startswith('align:'):
                align = stripped[len('align:'):].strip()
                if align not in ('start', 'center', 'end', 'stretch'):
                    raise self._error(
                        f"Invalid align '{align}'. Must be 'start', 'center', 'end', or 'stretch'.",
                        self.pos + 1,
                        nxt
                    )
                self._advance()
            elif stripped.startswith('justify:'):
                justify = stripped[len('justify:'):].strip()
                if justify not in ('start', 'center', 'end', 'space_between', 'space_around', 'space_evenly'):
                    raise self._error(
                        f"Invalid justify '{justify}'. Must be one of: start, center, end, space_between, space_around, space_evenly.",
                        self.pos + 1,
                        nxt
                    )
                self._advance()
            elif stripped.startswith('wrap:'):
                wrap_str = stripped[len('wrap:'):].strip().lower()
                wrap = wrap_str in ('true', 'yes', '1')
                self._advance()
            elif stripped.startswith('children:'):
                self._advance()
                # Use helper method to parse children with dash-list support
                children = self._parse_children_list(indent)
            elif stripped.startswith('style:'):
                self._advance()
                style = self._parse_kv_block(indent)
            elif stripped.startswith('layout:'):
                self._advance()
                layout = self._parse_layout_meta(indent)
            else:
                raise self._error(
                    f"Unknown stack property: {stripped.split(':')[0]}",
                    self.pos + 1,
                    nxt,
                    hint='Valid properties: direction, gap, align, justify, wrap, children, style, layout'
                )
        
        expanded_children = list(children)
        for child in children:
            if isinstance(child, ShowStatSummary) and getattr(child, "stats", None):
                for extra_stat in child.stats[1:]:
                    expanded_children.append(
                        ShowStatSummary(
                            label=extra_stat.get("label", child.label),
                            source_type=child.source_type,
                            source=child.source,
                            value=extra_stat.get("value", child.value),
                            format=extra_stat.get("format", child.format),
                            prefix=extra_stat.get("prefix", child.prefix),
                            suffix=extra_stat.get("suffix", child.suffix),
                            delta=extra_stat.get("delta", child.delta),
                            trend=extra_stat.get("trend", child.trend),
                            comparison_period=extra_stat.get("comparison_period", child.comparison_period),
                            sparkline=child.sparkline,
                            color=extra_stat.get("color", child.color),
                            icon=extra_stat.get("icon", child.icon),
                            layout=child.layout,
                            style=child.style,
                            stats=None,
                        )
                    )

        return StackLayout(
            direction=direction,
            gap=gap,
            align=align,
            justify=justify,
            wrap=wrap,
            children=expanded_children,
            style=style,
            layout=layout,
        )

    def _parse_layout_grid(self, line: str, base_indent: int) -> GridLayout:
        """
        Parse a grid layout component.
        
        Syntax: layout grid:
                    columns: 3
                    min_column_width: 200px
                    gap: large
                    responsive: true
                    children:
                        - show card "Card1" from dataset data1
                        - show card "Card2" from dataset data2
        """
        columns: Union[int, str] = "auto"
        min_column_width: Optional[str] = None
        gap: Union[str, int] = "medium"
        responsive = True
        children: List[Any] = []
        style: Optional[Dict[str, Any]] = None
        layout: Optional[LayoutMeta] = None
        
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
            
            if stripped.startswith('columns:'):
                col_str = stripped[len('columns:'):].strip()
                if col_str == "auto":
                    columns = "auto"
                else:
                    try:
                        columns = int(col_str)
                    except ValueError:
                        raise self._error(
                            f"Invalid columns value '{col_str}'. Must be an integer or 'auto'.",
                            self.pos + 1,
                            nxt
                        )
                self._advance()
            elif stripped.startswith('min_column_width:'):
                min_column_width = stripped[len('min_column_width:'):].strip()
                self._advance()
            elif stripped.startswith('gap:'):
                gap_str = stripped[len('gap:'):].strip()
                try:
                    gap = int(gap_str)
                except ValueError:
                    gap = gap_str
                self._advance()
            elif stripped.startswith('responsive:'):
                resp_str = stripped[len('responsive:'):].strip().lower()
                responsive = resp_str in ('true', 'yes', '1')
                self._advance()
            elif stripped.startswith('children:'):
                self._advance()
                # Use helper method to parse children with dash-list support
                children = self._parse_children_list(indent)
            elif stripped.startswith('style:'):
                self._advance()
                style = self._parse_kv_block(indent)
            elif stripped.startswith('layout:'):
                self._advance()
                layout = self._parse_layout_meta(indent)
            else:
                raise self._error(
                    f"Unknown grid property: {stripped.split(':')[0]}",
                    self.pos + 1,
                    nxt,
                    hint='Valid properties: columns, min_column_width, gap, responsive, children, style, layout'
                )
        
        return GridLayout(
            columns=columns,
            min_column_width=min_column_width,
            gap=gap,
            responsive=responsive,
            children=children,
            style=style,
            layout=layout,
        )

    def _parse_layout_split(self, line: str, base_indent: int) -> SplitLayout:
        """
        Parse a split layout component.
        
        Syntax: layout split:
                    ratio: 0.3
                    resizable: true
                    orientation: horizontal
                    left:
                        - show table "Orders" from dataset orders
                    right:
                        - show card "Details" from dataset details
        """
        ratio = 0.5
        resizable = False
        orientation = "horizontal"
        left: List[Any] = []
        right: List[Any] = []
        style: Optional[Dict[str, Any]] = None
        layout: Optional[LayoutMeta] = None
        
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
            
            if stripped.startswith('ratio:'):
                ratio_str = stripped[len('ratio:'):].strip()
                try:
                    ratio = float(ratio_str)
                    if not (0.0 <= ratio <= 1.0):
                        raise self._error(
                            f"Ratio must be between 0.0 and 1.0, got {ratio}",
                            self.pos + 1,
                            nxt
                        )
                except ValueError:
                    raise self._error(
                        f"Invalid ratio '{ratio_str}'. Must be a float between 0.0 and 1.0.",
                        self.pos + 1,
                        nxt
                    )
                self._advance()
            elif stripped.startswith('resizable:'):
                res_str = stripped[len('resizable:'):].strip().lower()
                resizable = res_str in ('true', 'yes', '1')
                self._advance()
            elif stripped.startswith('orientation:'):
                orientation = stripped[len('orientation:'):].strip()
                if orientation not in ('horizontal', 'vertical'):
                    raise self._error(
                        f"Invalid orientation '{orientation}'. Must be 'horizontal' or 'vertical'.",
                        self.pos + 1,
                        nxt
                    )
                self._advance()
            elif stripped.startswith('left:'):
                self._advance()
                # Use helper method to parse left children with dash-list support
                left = self._parse_children_list(indent)
            elif stripped.startswith('right:'):
                self._advance()
                # Use helper method to parse right children with dash-list support
                right = self._parse_children_list(indent)
            elif stripped.startswith('style:'):
                self._advance()
                style = self._parse_kv_block(indent)
            elif stripped.startswith('layout:'):
                self._advance()
                layout = self._parse_layout_meta(indent)
            else:
                raise self._error(
                    f"Unknown split property: {stripped.split(':')[0]}",
                    self.pos + 1,
                    nxt,
                    hint='Valid properties: ratio, resizable, orientation, left, right, style, layout'
                )
        
        return SplitLayout(
            left=left,
            right=right,
            ratio=ratio,
            resizable=resizable,
            orientation=orientation,
            style=style,
            layout=layout,
        )

    def _parse_layout_tabs(self, line: str, base_indent: int) -> TabsLayout:
        """
        Parse a tabs layout component.
        
        Supports two syntaxes:
        
        1. Verbose syntax:
           layout tabs:
               default_tab: overview
               persist_state: true
               tabs:
                   - id: overview
                     label: "Overview"
                     content:
                         - show card "Summary" from dataset summary
        
        2. Simple syntax (show tabs):
           show tabs:
               tab "Overview":
                   show card "Summary" from dataset summary
               tab "Details":
                   show text "Details content"
        """
        default_tab: Optional[str] = None
        persist_state = True
        tabs: List[TabItem] = []
        style: Optional[Dict[str, Any]] = None
        layout: Optional[LayoutMeta] = None
        
        # Check if using simple "show tabs" syntax
        using_simple_syntax = 'show tabs' in line
        
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
            
            # Simple syntax: tab "Name":
            if using_simple_syntax and stripped.startswith('tab '):
                tab_item = self._parse_simple_tab_item(line=stripped, base_indent=indent)
                tabs.append(tab_item)
                continue
            
            # Verbose syntax properties
            if stripped.startswith('default_tab:'):
                default_tab = stripped[len('default_tab:'):].strip()
                self._advance()
            elif stripped.startswith('persist_state:'):
                persist_str = stripped[len('persist_state:'):].strip().lower()
                persist_state = persist_str in ('true', 'yes', '1')
                self._advance()
            elif stripped.startswith('tabs:'):
                self._advance()
                # Parse tab items
                while self.pos < len(self.lines):
                    tab_line = self._peek()
                    if tab_line is None:
                        break
                    tab_indent = self._indent(tab_line)
                    tab_stripped = tab_line.strip()
                    if not tab_stripped or tab_stripped.startswith('#'):
                        self._advance()
                        continue
                    if tab_indent <= indent:
                        break
                    if tab_stripped.startswith('- '):
                        tab_item = self._parse_tab_item(tab_indent)
                        tabs.append(tab_item)
                    else:
                        break
            elif stripped.startswith('style:'):
                self._advance()
                style = self._parse_kv_block(indent)
            elif stripped.startswith('layout:'):
                self._advance()
                layout = self._parse_layout_meta(indent)
            else:
                raise self._error(
                    f"Unknown tabs property: {stripped.split(':')[0]}",
                    self.pos + 1,
                    nxt,
                    hint='Valid properties: default_tab, persist_state, tabs, style, layout (or use: tab "Name":)'
                )
        
        # Validate default_tab if specified
        if default_tab and tabs:
            tab_ids = [tab.id for tab in tabs]
            if default_tab not in tab_ids:
                raise self._error(
                    f"default_tab '{default_tab}' does not match any tab id. Available: {', '.join(tab_ids)}",
                    self.pos,
                    line
                )
        
        return TabsLayout(
            tabs=tabs,
            default_tab=default_tab,
            persist_state=persist_state,
            style=style,
            layout=layout,
        )

    def _parse_tab_item(self, base_indent: int) -> TabItem:
        """Parse a single tab item."""
        tab_id: Optional[str] = None
        label: Optional[str] = None
        icon: Optional[str] = None
        badge: Optional[Union[str, BadgeConfig]] = None
        content: List[Any] = []
        
        # First line starts with '- id:' or '- '
        line = self._peek()
        if line:
            stripped = line.strip()
            if stripped.startswith('- '):
                rest = stripped[2:].strip()
                if rest.startswith('id:'):
                    tab_id = rest[len('id:'):].strip()
                self._advance()
        
        # Parse nested properties
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent or stripped.startswith('- '):
                break
            
            if stripped.startswith('id:'):
                tab_id = stripped[len('id:'):].strip()
                self._advance()
            elif stripped.startswith('label:'):
                label = stripped[len('label:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('icon:'):
                icon = stripped[len('icon:'):].strip()
                self._advance()
            elif stripped.startswith('badge:'):
                badge = stripped[len('badge:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('content:'):
                self._advance()
                # Use helper method to parse content with dash-list support
                content = self._parse_children_list(indent)
            elif stripped.startswith('children:'):
                self._advance()
                content = self._parse_children_list(indent)
            else:
                break
        
        if not tab_id:
            raise self._error(
                "Tab item must have an 'id' property",
                self.pos,
                line or ""
            )
        if not label:
            raise self._error(
                f"Tab item '{tab_id}' must have a 'label' property",
                self.pos,
                line or ""
            )
        
        return TabItem(
            id=tab_id,
            label=label,
            icon=icon,
            badge=badge,
            content=content,
        )

    def _parse_simple_tab_item(self, line: str, base_indent: int) -> TabItem:
        """
        Parse a simple tab item.
        
        Syntax: tab "Name":
                    show card "..." from dataset ...
                    show text "..."
        """
        import re
        
        # Parse: tab "Name":
        match = re.match(r'tab\s+"([^"]+)"\s*:', line.strip())
        if not match:
            raise self._error(f'Expected: tab "Name":', self.pos + 1, line)
        
        label = match.group(1)
        tab_id = label.lower().replace(' ', '_')
        content = []
        
        self._advance()
        
        # Parse tab content (all statements until we hit another tab or outdent)
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            
            # Stop at outdent or next tab
            if indent <= base_indent:
                break
            if stripped.startswith('tab '):
                break
            
            # Parse page statement (show card, show text, etc.)
            stmt = self._parse_page_statement(indent)
            content.append(stmt)
        
        return TabItem(
            id=tab_id,
            label=label,
            icon=None,
            badge=None,
            content=content,
        )

    def _parse_layout_accordion(self, line: str, base_indent: int) -> AccordionLayout:
        """
        Parse an accordion layout component.
        
        Syntax: layout accordion:
                    multiple: true
                    items:
                        - id: section1
                          title: "Personal Information"
                          icon: user
                          default_open: true
                          content:
                              - show form "Profile" with fields name, email
        """
        multiple = False
        items: List[AccordionItem] = []
        style: Optional[Dict[str, Any]] = None
        layout: Optional[LayoutMeta] = None
        
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
            
            if stripped.startswith('multiple:'):
                mult_str = stripped[len('multiple:'):].strip().lower()
                multiple = mult_str in ('true', 'yes', '1')
                self._advance()
            elif stripped.startswith('allow_multiple:'):
                mult_str = stripped[len('allow_multiple:'):].strip().lower()
                multiple = mult_str in ('true', 'yes', '1')
                self._advance()
            elif stripped.startswith('items:') or stripped.startswith('sections:'):
                self._advance()
                # Parse accordion items
                while self.pos < len(self.lines):
                    item_line = self._peek()
                    if item_line is None:
                        break
                    item_indent = self._indent(item_line)
                    item_stripped = item_line.strip()
                    if not item_stripped or item_stripped.startswith('#'):
                        self._advance()
                        continue
                    if item_indent <= indent:
                        break
                    if item_stripped.startswith('- '):
                        accordion_item = self._parse_accordion_item(item_indent)
                        items.append(accordion_item)
                    else:
                        break
            elif stripped.startswith('style:'):
                self._advance()
                style = self._parse_kv_block(indent)
            elif stripped.startswith('layout:'):
                self._advance()
                layout = self._parse_layout_meta(indent)
            else:
                raise self._error(
                    f"Unknown accordion property: {stripped.split(':')[0]}",
                    self.pos + 1,
                    nxt,
                    hint='Valid properties: multiple, items, style, layout'
                )
        
        return AccordionLayout(
            items=items,
            multiple=multiple,
            style=style,
            layout=layout,
        )

    def _parse_accordion_item(self, base_indent: int) -> AccordionItem:
        """Parse a single accordion item."""
        item_id: Optional[str] = None
        title: Optional[str] = None
        description: Optional[str] = None
        icon: Optional[str] = None
        content: List[Any] = []
        default_open = False
        
        # First line starts with '- id:' or '- '
        line = self._peek()
        if line:
            stripped = line.strip()
            if stripped.startswith('- '):
                rest = stripped[2:].strip()
                if rest.startswith('id:'):
                    item_id = rest[len('id:'):].strip()
                self._advance()
        
        # Parse nested properties
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent or stripped.startswith('- '):
                break
            
            if stripped.startswith('id:'):
                item_id = stripped[len('id:'):].strip()
                self._advance()
            elif stripped.startswith('title:'):
                title = stripped[len('title:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('description:'):
                description = stripped[len('description:'):].strip().strip('"')
                self._advance()
            elif stripped.startswith('icon:'):
                icon = stripped[len('icon:'):].strip()
                self._advance()
            elif stripped.startswith('default_open:'):
                open_str = stripped[len('default_open:'):].strip().lower()
                default_open = open_str in ('true', 'yes', '1')
                self._advance()
            elif stripped.startswith('content:'):
                self._advance()
                # Use helper method to parse content with dash-list support
                content = self._parse_children_list(indent)
            elif stripped.startswith('children:'):
                self._advance()
                content = self._parse_children_list(indent)
            else:
                break
        
        if not item_id:
            raise self._error(
                "Accordion item must have an 'id' property",
                self.pos,
                line or ""
            )
        if not title:
            raise self._error(
                f"Accordion item '{item_id}' must have a 'title' property",
                self.pos,
                line or ""
            )
        
        return AccordionItem(
            id=item_id,
            title=title,
            description=description,
            icon=icon,
            content=content,
            default_open=default_open,
        )

    def _parse_layout_meta(self, base_indent: int) -> LayoutMeta:
        """Parse layout metadata block."""
        direction: Optional[str] = None
        spacing: Optional[str] = None
        width: Optional[int] = None
        height: Optional[int] = None
        variant: Optional[str] = None
        align: Optional[str] = None
        emphasis: Optional[str] = None
        extras: Dict[str, Any] = {}
        
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
            
            if ':' in stripped:
                key, value = stripped.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'direction':
                    direction = value
                elif key == 'spacing':
                    spacing = value
                elif key == 'width':
                    try:
                        width = int(value)
                    except ValueError:
                        pass
                elif key == 'height':
                    try:
                        height = int(value)
                    except ValueError:
                        pass
                elif key == 'variant':
                    variant = value
                elif key == 'align':
                    align = value
                elif key == 'emphasis':
                    emphasis = value
                else:
                    extras[key] = value
                
                self._advance()
            else:
                break
        
        return LayoutMeta(
            direction=direction,
            spacing=spacing,
            width=width,
            height=height,
            variant=variant,
            align=align,
            emphasis=emphasis,
            extras=extras,
        )

    # ==========================================================================
    # NAVIGATION & APP CHROME PARSING METHODS
    # ==========================================================================

    def _parse_sidebar(self, line: str, base_indent: int) -> Sidebar:
        """
        Parse sidebar navigation component.
        
        Syntax: sidebar:
            item "Home" at "/home" icon "home"
            item "Dashboard" at "/dashboard" icon "chart"
            section "Settings":
                item "Profile" at "/profile"
                item "Security" at "/security"
            collapsible: true
            width: normal
            position: left
        """
        if not line.strip().endswith(':'):
            raise self._error(
                "Sidebar declaration must end with ':'",
                self.pos,
                line,
                hint='Syntax: sidebar:'
            )
        
        items: List[NavItem] = []
        sections: List[NavSection] = []
        collapsible = False
        collapsed_by_default = False
        width: Optional[str] = None
        position = "left"
        
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
            
            self._advance()
            
            if stripped.startswith('item '):
                item = self._parse_nav_item(stripped, indent)
                items.append(item)
            elif stripped.startswith('section '):
                section = self._parse_nav_section(stripped, indent)
                sections.append(section)
            elif stripped.startswith('collapsible:'):
                collapsible = self._coerce_bool_with_context(
                    stripped.split(':', 1)[1].strip(),
                    'collapsible',
                    self.pos,
                    nxt
                )
            elif stripped.startswith('collapsed by default:'):
                collapsed_by_default = self._coerce_bool_with_context(
                    stripped.split(':', 1)[1].strip(),
                    'collapsed by default',
                    self.pos,
                    nxt
                )
            elif stripped.startswith('width:'):
                width = stripped.split(':', 1)[1].strip()
            elif stripped.startswith('position:'):
                position = stripped.split(':', 1)[1].strip()
            else:
                raise self._error(
                    f"Unknown sidebar property: {stripped.split(':')[0]}",
                    self.pos,
                    nxt,
                    hint='Valid properties: item, section, collapsible, collapsed by default, width, position'
                )
        
        return Sidebar(
            items=items,
            sections=sections,
            collapsible=collapsible,
            collapsed_by_default=collapsed_by_default,
            width=width,
            position=position
        )

    def _parse_nav_item(self, line: str, base_indent: int) -> NavItem:
        """Parse a single navigation item."""
        # item "Label" at "route" icon "icon-name" badge {count: 5} action "action-id"
        match = re.match(
            r'item\s+"([^"]+)"'  # label (required)
            r'(?:\s+at\s+"([^"]+)")?'  # route (optional)
            r'(?:\s+icon\s+"([^"]+)")?'  # icon (optional)
            r'(?:\s+badge\s+(\{[^}]+\}))?'  # badge dict (optional)
            r'(?:\s+action\s+"([^"]+)")?',  # action (optional)
            line
        )
        if not match:
            raise self._error(
                'Invalid nav item syntax',
                self.pos,
                line,
                hint='Syntax: item "Label" [at "route"] [icon "name"] [badge {...}] [action "id"]'
            )
        
        label = match.group(1)
        route = match.group(2)
        icon = match.group(3)
        badge_str = match.group(4)
        action = match.group(5)
        
        badge: Optional[Dict[str, Any]] = None
        if badge_str:
            # Parse badge dict: {text: "value"} or {count: 5}
            badge_str = badge_str.strip()
            if badge_str.startswith('{') and badge_str.endswith('}'):
                # Extract key:value from {key: value}
                inner = badge_str[1:-1].strip()
                if ':' in inner:
                    key, val = inner.split(':', 1)
                    key = key.strip()
                    val = val.strip().strip('"')  # Remove quotes if present
                    try:
                        # Try to parse as number
                        val = int(val)
                    except ValueError:
                        pass  # Keep as string
                    badge = {key: val}
                else:
                    badge = {'text': inner}
            else:
                badge = {'text': badge_str}
        
        children: List[NavItem] = []
        # Check for nested items
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            if indent <= base_indent:
                break
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if stripped.startswith('item '):
                self._advance()
                child = self._parse_nav_item(stripped, indent)
                children.append(child)
            else:
                break
        
        return NavItem(
            id=label.lower().replace(' ', '_'),
            label=label,
            route=route,
            icon=icon,
            badge=badge,
            action=action,
            children=children
        )

    def _parse_nav_section(self, line: str, base_indent: int) -> NavSection:
        """Parse a navigation section grouping."""
        # section "Label":
        match = re.match(r'section\s+"([^"]+)"\s*:', line)
        if not match:
            raise self._error(
                'Invalid nav section syntax',
                self.pos,
                line,
                hint='Syntax: section "Label":'
            )
        
        label = match.group(1)
        item_ids: List[str] = []
        collapsible = False
        collapsed_by_default = False
        
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
            
            self._advance()
            
            if stripped.startswith('item '):
                # For now, just extract the ID from the label
                item_match = re.match(r'item\s+"([^"]+)"', stripped)
                if item_match:
                    item_label = item_match.group(1)
                    item_ids.append(item_label.lower().replace(' ', '_'))
            elif stripped.startswith('collapsible:'):
                collapsible = self._coerce_bool_with_context(
                    stripped.split(':', 1)[1].strip(),
                    'collapsible',
                    self.pos,
                    nxt
                )
            elif stripped.startswith('collapsed by default:'):
                collapsed_by_default = self._coerce_bool_with_context(
                    stripped.split(':', 1)[1].strip(),
                    'collapsed by default',
                    self.pos,
                    nxt
                )
        
        return NavSection(
            id=label.lower().replace(' ', '_'),
            label=label,
            items=item_ids,
            collapsible=collapsible,
            collapsed_by_default=collapsed_by_default
        )

    def _parse_navbar(self, line: str, base_indent: int) -> Navbar:
        """
        Parse navbar/topbar component.
        
        Syntax: navbar:
            logo: "assets/logo.png"
            title: "My App"
            action "User Menu" icon "user" type "menu":
                item "Profile" at "/profile"
                item "Logout" action "logout"
            position: top
            sticky: true
        """
        if not line.strip().endswith(':'):
            raise self._error(
                "Navbar declaration must end with ':'",
                self.pos,
                line,
                hint='Syntax: navbar:'
            )
        
        logo: Optional[str] = None
        title: Optional[str] = None
        actions: List[NavbarAction] = []
        position = "top"
        sticky = True
        
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
            
            self._advance()
            
            if stripped.startswith('logo:'):
                logo = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('title:'):
                title = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('action '):
                action = self._parse_navbar_action(stripped, indent)
                actions.append(action)
            elif stripped.startswith('position:'):
                position = stripped.split(':', 1)[1].strip()
            elif stripped.startswith('sticky:'):
                sticky = self._coerce_bool_with_context(
                    stripped.split(':', 1)[1].strip(),
                    'sticky',
                    self.pos,
                    nxt
                )
            else:
                raise self._error(
                    f"Unknown navbar property: {stripped.split(':')[0]}",
                    self.pos,
                    nxt,
                    hint='Valid properties: logo, title, action, position, sticky'
                )
        
        return Navbar(
            logo=logo,
            title=title,
            actions=actions,
            position=position,
            sticky=sticky
        )

    def _parse_navbar_action(self, line: str, base_indent: int) -> NavbarAction:
        """Parse a navbar action button/menu."""
        # action "Label" icon "icon-name" type "button"|"menu"|"toggle"[:]
        match = re.match(r'action\s+"([^"]+)"(?:\s+icon\s+"([^"]+)")?(?:\s+type\s+"([^"]+)")?\s*:?', line)
        if not match:
            raise self._error(
                'Invalid navbar action syntax',
                self.pos,
                line,
                hint='Syntax: action "Label" [icon "name"] [type "button|menu|toggle"][:]'
            )
        
        label = match.group(1)
        icon = match.group(2)
        action_type = match.group(3) or "button"
        
        menu_items: List[NavItem] = []
        action_id: Optional[str] = None
        
        # Parse nested menu items or action reference
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
            
            self._advance()
            
            if stripped.startswith('item '):
                item = self._parse_nav_item(stripped, indent)
                menu_items.append(item)
            elif stripped.startswith('action:'):
                action_id = stripped.split(':', 1)[1].strip()
            else:
                break
        
        return NavbarAction(
            id=label.lower().replace(' ', '_'),
            label=label,
            icon=icon,
            type=action_type,
            action=action_id,
            menu_items=menu_items
        )

    def _parse_breadcrumbs(self, line: str, base_indent: int) -> Breadcrumbs:
        """
        Parse breadcrumbs component.
        
        Syntax: breadcrumbs:
            item "Home" at "/"
            item "Dashboard" at "/dashboard"
            item "Settings"
            auto derive: true
            separator: "/"
        """
        if not line.strip().endswith(':'):
            raise self._error(
                "Breadcrumbs declaration must end with ':'",
                self.pos,
                line,
                hint='Syntax: breadcrumbs:'
            )
        
        items: List[BreadcrumbItem] = []
        auto_derive = False
        separator = "/"
        
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
            
            self._advance()
            
            if stripped.startswith('item '):
                item = self._parse_breadcrumb_item(stripped)
                items.append(item)
            elif stripped.startswith('auto derive:') or stripped.startswith('auto_derive:'):
                auto_derive = self._coerce_bool_with_context(
                    stripped.split(':', 1)[1].strip(),
                    'auto_derive',
                    self.pos,
                    nxt
                )
            elif stripped.startswith('separator:'):
                separator = stripped.split(':', 1)[1].strip().strip('"')
            else:
                raise self._error(
                    f"Unknown breadcrumbs property: {stripped.split(':')[0]}",
                    self.pos,
                    nxt,
                    hint='Valid properties: item, auto_derive (or auto derive), separator'
                )
        
        return Breadcrumbs(
            items=items,
            auto_derive=auto_derive,
            separator=separator
        )

    def _parse_breadcrumb_item(self, line: str) -> BreadcrumbItem:
        """Parse a single breadcrumb item."""
        # item "Label" [at "route"]
        match = re.match(r'item\s+"([^"]+)"(?:\s+at\s+"([^"]+)")?', line)
        if not match:
            raise self._error(
                'Invalid breadcrumb item syntax',
                self.pos,
                line,
                hint='Syntax: item "Label" [at "route"]'
            )
        
        label = match.group(1)
        route = match.group(2)
        
        return BreadcrumbItem(label=label, route=route)

    def _parse_command_palette(self, line: str, base_indent: int) -> CommandPalette:
        """
        Parse command palette component.
        
        Syntax: command palette:
            shortcut: "ctrl+k"
            source routes
            source actions
            source custom:
                item "Command 1" action "cmd1"
            placeholder: "Search commands..."
            max results: 10
        """
        if not line.strip().endswith(':'):
            raise self._error(
                "Command palette declaration must end with ':'",
                self.pos,
                line,
                hint='Syntax: command palette:'
            )
        
        shortcut = "ctrl+k"
        sources: List[CommandSource] = []
        placeholder = "Search commands..."
        max_results = 10
        
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
            
            self._advance()
            
            if stripped.startswith('shortcut:'):
                shortcut = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('source '):
                source = self._parse_command_source(stripped, indent)
                sources.append(source)
            elif stripped.startswith('placeholder:'):
                placeholder = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('max results:'):
                max_results = int(stripped.split(':', 1)[1].strip())
            else:
                raise self._error(
                    f"Unknown command palette property: {stripped.split(':')[0]}",
                    self.pos,
                    nxt,
                    hint='Valid properties: shortcut, source, placeholder, max results'
                )
        
        return CommandPalette(
            shortcut=shortcut,
            sources=sources,
            placeholder=placeholder,
            max_results=max_results
        )

    def _parse_command_source(self, line: str, base_indent: int) -> CommandSource:
        """Parse a command palette source."""
        # source routes
        # source actions
        # source "id" from "endpoint" label "label"  (API-backed)
        # source custom:
        #     item "Command" action "cmd_id"
        
        # Try API-backed source first: source "id" from "endpoint" label "label"
        api_match = re.match(r'source\s+"([^"]+)"\s+from\s+"([^"]+)"(?:\s+label\s+"([^"]+)")?', line)
        if api_match:
            source_id = api_match.group(1)
            endpoint = api_match.group(2)
            label = api_match.group(3) or source_id.capitalize()
            
            return CommandSource(
                type='api',
                id=source_id,
                endpoint=endpoint,
                label=label,
                custom_items=[]
            )
        
        # Try standard source types
        match = re.match(r'source\s+(routes|actions|custom)\s*:?', line)
        if not match:
            raise self._error(
                'Invalid command source syntax',
                self.pos,
                line,
                hint='Syntax: source routes|actions|custom or source "id" from "endpoint" label "label"'
            )
        
        source_type = match.group(1)
        custom_items: List[Dict[str, Any]] = []
        
        if line.endswith(':'):
            # Parse custom items
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
                
                self._advance()
                
                if stripped.startswith('item '):
                    # item "Label" action "action_id"
                    item_match = re.match(r'item\s+"([^"]+)"(?:\s+action\s+"([^"]+)")?', stripped)
                    if item_match:
                        custom_items.append({
                            'label': item_match.group(1),
                            'action': item_match.group(2)
                        })
                else:
                    break
        
        return CommandSource(
            type=source_type,
            custom_items=custom_items
        )

    # ============================================================
    # FEEDBACK COMPONENTS (Modal, Toast)
    # ============================================================

    def parse_modal(self, line: str) -> Modal:
        """
        Parse modal component.
        
        Syntax: modal "modal_id":
                  title: "Confirm Delete"
                  description: "Are you sure?"
                  size: "md"
                  dismissible: true
                  trigger: "open_confirm"
                  content:
                    show text "Content goes here"
                  actions:
                    action "Cancel" variant "ghost"
                    action "Delete" variant "destructive" action "delete_item"
        """
        if not line.strip().endswith(':'):
            raise self._error(
                "Modal declaration must end with ':'",
                self.pos,
                line,
                hint='Syntax: modal "modal_id":'
            )
        
        # Extract modal ID (quoted or unquoted)
        match = re.match(r'\s*modal\s+(?:"([^"]+)"|([a-zA-Z_][a-zA-Z0-9_]*))\s*:', line)
        if not match:
            raise self._error(
                'Modal requires an ID',
                self.pos,
                line,
                hint='Syntax: modal "modal_id": or modal modal_id:'
            )
        
        modal_id = match.group(1) or match.group(2)
        base_indent = self._indent(line)
        
        title = ""
        description = None
        size = "md"
        dismissible = True
        trigger = None
        content: List[Statement] = []
        actions: List[ModalAction] = []
        variant = None
        tone = None
        modal_size = None
        
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
            
            self._advance()
            
            if stripped.startswith('title:'):
                title = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('description:'):
                description = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('size:'):
                size = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('dismissible:'):
                dismissible = self._coerce_bool_with_context(
                    stripped.split(':', 1)[1].strip(),
                    'dismissible',
                    self.pos,
                    nxt
                )
            elif stripped.startswith('trigger:'):
                trigger = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('variant:'):
                value = stripped.split(':', 1)[1].strip().strip('"')
                variant = self._parse_design_token(value, "variant", self.pos, nxt)
            elif stripped.startswith('tone:'):
                value = stripped.split(':', 1)[1].strip().strip('"')
                tone = self._parse_design_token(value, "tone", self.pos, nxt)
            elif stripped.startswith('modal_size:'):
                value = stripped.split(':', 1)[1].strip().strip('"')
                modal_size = self._parse_design_token(value, "size", self.pos, nxt)
            elif stripped.startswith('content:'):
                # Parse nested statements using helper
                content = self._parse_children_list(indent)
            elif stripped.startswith('actions:'):
                # Parse modal actions
                actions_indent = indent
                while self.pos < len(self.lines):
                    action_line = self._peek()
                    if action_line is None:
                        break
                    action_indent = self._indent(action_line)
                    action_stripped = action_line.strip()
                    
                    if not action_stripped or action_stripped.startswith('#'):
                        self._advance()
                        continue
                    if action_indent <= actions_indent:
                        break
                    
                    self._advance()
                    
                    if action_stripped.startswith('action '):
                        action = self._parse_modal_action(action_stripped)
                        actions.append(action)
                    else:
                        break
            else:
                raise self._error(
                    f"Unknown modal property: {stripped.split(':')[0]}",
                    self.pos - 1,
                    stripped,
                    hint='Valid properties: title, description, size, dismissible, trigger, content, actions'
                )
        
        return Modal(
            id=modal_id,
            title=title,
            description=description,
            content=content,
            actions=actions,
            size=size,
            dismissible=dismissible,
            trigger=trigger,
            variant=variant,
            tone=tone,
            modal_size=modal_size,
        )
    
    def _parse_modal_action(self, line: str) -> ModalAction:
        """Parse a modal action button."""
        # action "Label" [variant "primary"] [action "action_id"] [close: false]
        match = re.match(r'action\s+"([^"]+)"', line)
        if not match:
            raise self._error(
                'Modal action requires a label',
                self.pos - 1,
                line,
                hint='Syntax: action "Label" [variant "primary"] [action "action_id"]'
            )
        
        label = match.group(1)
        variant = "default"
        action_name = None
        close = True
        
        # Extract optional parameters after the label
        # Use findall to get all matches, then filter
        remaining_line = line[match.end():]
        
        # Extract variant
        variant_match = re.search(r'variant\s+"([^"]+)"', remaining_line)
        if variant_match:
            variant = variant_match.group(1)
        
        # Extract action identifier (different from the 'action' keyword at start)
        action_match = re.search(r'action\s+"([^"]+)"', remaining_line)
        if action_match:
            action_name = action_match.group(1)
        
        # Extract close flag
        close_match = re.search(r'close:\s*(true|false)', remaining_line)
        if close_match:
            close = close_match.group(1) == 'true'
        
        return ModalAction(
            label=label,
            action=action_name,
            variant=variant,
            close=close
        )

    def parse_toast(self, line: str) -> Toast:
        """
        Parse toast notification component.
        
        Syntax: toast "toast_id":
                  title: "Success"
                  description: "Item saved successfully"
                  variant: "success"
                  duration: 3000
                  position: "top-right"
                  trigger: "show_success"
                  action_label: "Undo"
                  action: "undo_save"
        """
        if not line.strip().endswith(':'):
            raise self._error(
                "Toast declaration must end with ':'",
                self.pos,
                line,
                hint='Syntax: toast "toast_id":'
            )
        
        # Extract toast ID (quoted or unquoted)
        match = re.match(r'\s*toast\s+(?:"([^"]+)"|([a-zA-Z_][a-zA-Z0-9_]*))\s*:', line)
        if not match:
            raise self._error(
                'Toast requires an ID',
                self.pos,
                line,
                hint='Syntax: toast "toast_id": or toast toast_id:'
            )
        
        toast_id = match.group(1) or match.group(2)
        base_indent = self._indent(line)
        
        title = ""
        description = None
        variant = "default"
        duration = 3000
        action_label = None
        action_name = None
        position = "top-right"
        trigger = None
        toast_variant = None
        tone = None
        size = None
        
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
            
            self._advance()
            
            if stripped.startswith('title:'):
                title = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('description:'):
                description = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('variant:'):
                variant = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('duration:'):
                duration = int(stripped.split(':', 1)[1].strip())
            elif stripped.startswith('action_label:'):
                action_label = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('action:'):
                action_name = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('position:'):
                position = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('trigger:'):
                trigger = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('toast_variant:'):
                value = stripped.split(':', 1)[1].strip().strip('"')
                toast_variant = self._parse_design_token(value, "variant", self.pos, nxt)
            elif stripped.startswith('tone:'):
                value = stripped.split(':', 1)[1].strip().strip('"')
                tone = self._parse_design_token(value, "tone", self.pos, nxt)
            elif stripped.startswith('size:'):
                value = stripped.split(':', 1)[1].strip().strip('"')
                size = self._parse_design_token(value, "size", self.pos, nxt)
            else:
                raise self._error(
                    f"Unknown toast property: {stripped.split(':')[0]}",
                    self.pos - 1,
                    stripped,
                    hint='Valid properties: title, description, variant, duration, action_label, action, position, trigger, toast_variant, tone, size'
                )
        
        return Toast(
            id=toast_id,
            title=title,
            description=description,
            variant=variant,
            duration=duration,
            action_label=action_label,
            action=action_name,
            position=position,
            trigger=trigger,
            toast_variant=toast_variant,
            tone=tone,
            size=size,
        )

    # =============================================================================
    # AI Semantic Component Parsers
    # =============================================================================

    def parse_chat_thread(self, line: str) -> 'ChatThread':
        """
        Parse chat thread component for AI conversations.
        
        Syntax: chat_thread "thread_id":
                  messages_binding: "conversation.messages"
                  group_by: "role"
                  show_timestamps: true
                  streaming_enabled: true
                  streaming_source: "agent.stream"
        """
        from namel3ss.ast.pages import ChatThread
        
        if not line.strip().endswith(':'):
            raise self._error(
                "Chat thread declaration must end with ':'",
                self.pos,
                line,
                hint='Syntax: chat_thread "thread_id":'
            )
        
        # Extract thread ID
        match = re.match(r'\s*chat_thread\s+(?:"([^"]+)"|([a-zA-Z_][a-zA-Z0-9_]*))\s*:', line)
        if not match:
            raise self._error(
                'Chat thread requires an ID',
                self.pos,
                line,
                hint='Syntax: chat_thread "thread_id":'
            )
        
        thread_id = match.group(1) or match.group(2)
        base_indent = self._indent(line)
        
        # Default values
        messages_binding = ""
        group_by = "role"
        show_timestamps = True
        show_avatar = True
        reverse_order = False
        auto_scroll = True
        max_height = None
        streaming_enabled = False
        streaming_source = None
        show_role_labels = True
        show_token_count = False
        enable_copy = True
        enable_regenerate = False
        variant = "default"
        
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
            
            self._advance()
            
            if stripped.startswith('messages_binding:'):
                messages_binding = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('group_by:'):
                group_by = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('show_timestamps:'):
                show_timestamps = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_avatar:'):
                show_avatar = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('reverse_order:'):
                reverse_order = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('auto_scroll:'):
                auto_scroll = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('max_height:'):
                max_height = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('streaming_enabled:'):
                streaming_enabled = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('streaming_source:'):
                streaming_source = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('show_role_labels:'):
                show_role_labels = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_token_count:'):
                show_token_count = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('enable_copy:'):
                enable_copy = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('enable_regenerate:'):
                enable_regenerate = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('variant:'):
                variant = stripped.split(':', 1)[1].strip().strip('"')
        
        if not messages_binding:
            raise self._error(
                "Chat thread requires 'messages_binding' property",
                self.pos - 1,
                line,
                hint="Example: messages_binding: \"conversation.messages\""
            )
        
        return ChatThread(
            id=thread_id,
            messages_binding=messages_binding,
            group_by=group_by,
            show_timestamps=show_timestamps,
            show_avatar=show_avatar,
            reverse_order=reverse_order,
            auto_scroll=auto_scroll,
            max_height=max_height,
            streaming_enabled=streaming_enabled,
            streaming_source=streaming_source,
            show_role_labels=show_role_labels,
            show_token_count=show_token_count,
            enable_copy=enable_copy,
            enable_regenerate=enable_regenerate,
            variant=variant
        )

    def parse_agent_panel(self, line: str) -> 'AgentPanel':
        """
        Parse agent panel component for displaying agent state and metrics.
        
        Syntax: agent_panel "panel_id":
                  agent_binding: "current_agent"
                  metrics_binding: "run.metrics"
                  show_status: true
                  show_tokens: true
        """
        from namel3ss.ast.pages import AgentPanel
        
        if not line.strip().endswith(':'):
            raise self._error(
                "Agent panel declaration must end with ':'",
                self.pos,
                line,
                hint='Syntax: agent_panel "panel_id":'
            )
        
        # Extract panel ID
        match = re.match(r'\s*agent_panel\s+(?:"([^"]+)"|([a-zA-Z_][a-zA-Z0-9_]*))\s*:', line)
        if not match:
            raise self._error(
                'Agent panel requires an ID',
                self.pos,
                line,
                hint='Syntax: agent_panel "panel_id":'
            )
        
        panel_id = match.group(1) or match.group(2)
        base_indent = self._indent(line)
        
        # Default values
        agent_binding = ""
        metrics_binding = None
        show_status = True
        show_metrics = True
        show_profile = False
        show_limits = False
        show_last_error = False
        show_tools = False
        show_tokens = True
        show_cost = True
        show_latency = True
        show_model = True
        variant = "card"
        compact = False
        
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
            
            self._advance()
            
            if stripped.startswith('agent_binding:'):
                agent_binding = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('metrics_binding:'):
                metrics_binding = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('show_status:'):
                show_status = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_metrics:'):
                show_metrics = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_profile:'):
                show_profile = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_limits:'):
                show_limits = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_last_error:'):
                show_last_error = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_tools:'):
                show_tools = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_tokens:'):
                show_tokens = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_cost:'):
                show_cost = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_latency:'):
                show_latency = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_model:'):
                show_model = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('variant:'):
                variant = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('compact:'):
                compact = stripped.split(':', 1)[1].strip().lower() == 'true'
        
        if not agent_binding:
            raise self._error(
                "Agent panel requires 'agent_binding' property",
                self.pos - 1,
                line,
                hint="Example: agent_binding: \"current_agent\""
            )
        
        return AgentPanel(
            id=panel_id,
            agent_binding=agent_binding,
            metrics_binding=metrics_binding,
            show_status=show_status,
            show_metrics=show_metrics,
            show_profile=show_profile,
            show_limits=show_limits,
            show_last_error=show_last_error,
            show_tools=show_tools,
            show_tokens=show_tokens,
            show_cost=show_cost,
            show_latency=show_latency,
            show_model=show_model,
            variant=variant,
            compact=compact
        )

    def parse_tool_call_view(self, line: str) -> 'ToolCallView':
        """
        Parse tool call view component for displaying tool invocations.
        
        Syntax: tool_call_view "view_id":
                  calls_binding: "run.tool_calls"
                  show_inputs: true
                  show_outputs: true
                  variant: "list"
        """
        from namel3ss.ast.pages import ToolCallView
        
        if not line.strip().endswith(':'):
            raise self._error(
                "Tool call view declaration must end with ':'",
                self.pos,
                line,
                hint='Syntax: tool_call_view "view_id":'
            )
        
        # Extract view ID
        match = re.match(r'\s*tool_call_view\s+(?:"([^"]+)"|([a-zA-Z_][a-zA-Z0-9_]*))\s*:', line)
        if not match:
            raise self._error(
                'Tool call view requires an ID',
                self.pos,
                line,
                hint='Syntax: tool_call_view "view_id":'
            )
        
        view_id = match.group(1) or match.group(2)
        base_indent = self._indent(line)
        
        # Default values
        calls_binding = ""
        show_inputs = True
        show_outputs = True
        show_timing = True
        show_status = True
        show_raw_payload = False
        filter_tool_name = None
        filter_status = None
        variant = "list"
        expandable = True
        max_height = None
        enable_retry = False
        enable_copy = True
        
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
            
            self._advance()
            
            if stripped.startswith('calls_binding:'):
                calls_binding = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('show_inputs:'):
                show_inputs = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_outputs:'):
                show_outputs = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_timing:'):
                show_timing = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_status:'):
                show_status = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_raw_payload:'):
                show_raw_payload = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('variant:'):
                variant = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('expandable:'):
                expandable = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('max_height:'):
                max_height = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('enable_retry:'):
                enable_retry = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('enable_copy:'):
                enable_copy = stripped.split(':', 1)[1].strip().lower() == 'true'
        
        if not calls_binding:
            raise self._error(
                "Tool call view requires 'calls_binding' property",
                self.pos - 1,
                line,
                hint="Example: calls_binding: \"run.tool_calls\""
            )
        
        return ToolCallView(
            id=view_id,
            calls_binding=calls_binding,
            show_inputs=show_inputs,
            show_outputs=show_outputs,
            show_timing=show_timing,
            show_status=show_status,
            show_raw_payload=show_raw_payload,
            filter_tool_name=filter_tool_name,
            filter_status=filter_status,
            variant=variant,
            expandable=expandable,
            max_height=max_height,
            enable_retry=enable_retry,
            enable_copy=enable_copy
        )

    def parse_log_view(self, line: str) -> 'LogView':
        """
        Parse log view component for displaying logs and traces.
        
        Syntax: log_view "view_id":
                  logs_binding: "run.logs"
                  level_filter: ["info", "warn", "error"]
                  search_enabled: true
                  auto_scroll: true
        """
        from namel3ss.ast.pages import LogView
        
        if not line.strip().endswith(':'):
            raise self._error(
                "Log view declaration must end with ':'",
                self.pos,
                line,
                hint='Syntax: log_view "view_id":'
            )
        
        # Extract view ID
        match = re.match(r'\s*log_view\s+(?:"([^"]+)"|([a-zA-Z_][a-zA-Z0-9_]*))\s*:', line)
        if not match:
            raise self._error(
                'Log view requires an ID',
                self.pos,
                line,
                hint='Syntax: log_view "view_id":'
            )
        
        view_id = match.group(1) or match.group(2)
        base_indent = self._indent(line)
        
        # Default values
        logs_binding = ""
        level_filter = None
        search_enabled = True
        search_placeholder = "Search logs..."
        show_timestamp = True
        show_level = True
        show_metadata = False
        show_source = False
        auto_scroll = True
        auto_refresh = False
        refresh_interval = 5000
        max_entries = 1000
        variant = "default"
        max_height = None
        virtualized = True
        enable_copy = True
        enable_download = False
        
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
            
            self._advance()
            
            if stripped.startswith('logs_binding:'):
                logs_binding = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('search_enabled:'):
                search_enabled = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('search_placeholder:'):
                search_placeholder = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('show_timestamp:'):
                show_timestamp = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_level:'):
                show_level = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_metadata:'):
                show_metadata = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_source:'):
                show_source = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('auto_scroll:'):
                auto_scroll = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('auto_refresh:'):
                auto_refresh = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('refresh_interval:'):
                refresh_interval = int(stripped.split(':', 1)[1].strip())
            elif stripped.startswith('max_entries:'):
                max_entries = int(stripped.split(':', 1)[1].strip())
            elif stripped.startswith('variant:'):
                variant = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('max_height:'):
                max_height = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('virtualized:'):
                virtualized = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('enable_copy:'):
                enable_copy = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('enable_download:'):
                enable_download = stripped.split(':', 1)[1].strip().lower() == 'true'
        
        if not logs_binding:
            raise self._error(
                "Log view requires 'logs_binding' property",
                self.pos - 1,
                line,
                hint="Example: logs_binding: \"run.logs\""
            )
        
        return LogView(
            id=view_id,
            logs_binding=logs_binding,
            level_filter=level_filter,
            search_enabled=search_enabled,
            search_placeholder=search_placeholder,
            show_timestamp=show_timestamp,
            show_level=show_level,
            show_metadata=show_metadata,
            show_source=show_source,
            auto_scroll=auto_scroll,
            auto_refresh=auto_refresh,
            refresh_interval=refresh_interval,
            max_entries=max_entries,
            variant=variant,
            max_height=max_height,
            virtualized=virtualized,
            enable_copy=enable_copy,
            enable_download=enable_download
        )

    def parse_evaluation_result(self, line: str) -> 'EvaluationResult':
        """
        Parse evaluation result component for displaying eval metrics.
        
        Syntax: evaluation_result "result_id":
                  eval_run_binding: "eval.run_123"
                  show_summary: true
                  show_histograms: true
                  primary_metric: "accuracy"
        """
        from namel3ss.ast.pages import EvaluationResult
        
        if not line.strip().endswith(':'):
            raise self._error(
                "Evaluation result declaration must end with ':'",
                self.pos,
                line,
                hint='Syntax: evaluation_result "result_id":'
            )
        
        # Extract result ID
        match = re.match(r'\s*evaluation_result\s+(?:"([^"]+)"|([a-zA-Z_][a-zA-Z0-9_]*))\s*:', line)
        if not match:
            raise self._error(
                'Evaluation result requires an ID',
                self.pos,
                line,
                hint='Syntax: evaluation_result "result_id":'
            )
        
        result_id = match.group(1) or match.group(2)
        base_indent = self._indent(line)
        
        # Default values
        eval_run_binding = ""
        show_summary = True
        show_histograms = True
        show_error_table = True
        show_metadata = False
        metrics_to_show = None
        primary_metric = None
        filter_metric = None
        filter_min_score = None
        filter_max_score = None
        filter_status = None
        show_error_distribution = True
        show_error_examples = True
        max_error_examples = 10
        variant = "dashboard"
        comparison_run_binding = None
        
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
            
            self._advance()
            
            if stripped.startswith('eval_run_binding:'):
                eval_run_binding = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('show_summary:'):
                show_summary = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_histograms:'):
                show_histograms = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_error_table:'):
                show_error_table = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_metadata:'):
                show_metadata = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('primary_metric:'):
                primary_metric = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('filter_metric:'):
                filter_metric = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('filter_min_score:'):
                filter_min_score = float(stripped.split(':', 1)[1].strip())
            elif stripped.startswith('filter_max_score:'):
                filter_max_score = float(stripped.split(':', 1)[1].strip())
            elif stripped.startswith('show_error_distribution:'):
                show_error_distribution = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_error_examples:'):
                show_error_examples = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('max_error_examples:'):
                max_error_examples = int(stripped.split(':', 1)[1].strip())
            elif stripped.startswith('variant:'):
                variant = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('comparison_run_binding:'):
                comparison_run_binding = stripped.split(':', 1)[1].strip().strip('"')
        
        if not eval_run_binding:
            raise self._error(
                "Evaluation result requires 'eval_run_binding' property",
                self.pos - 1,
                line,
                hint="Example: eval_run_binding: \"eval.run_123\""
            )
        
        return EvaluationResult(
            id=result_id,
            eval_run_binding=eval_run_binding,
            show_summary=show_summary,
            show_histograms=show_histograms,
            show_error_table=show_error_table,
            show_metadata=show_metadata,
            metrics_to_show=metrics_to_show,
            primary_metric=primary_metric,
            filter_metric=filter_metric,
            filter_min_score=filter_min_score,
            filter_max_score=filter_max_score,
            filter_status=filter_status,
            show_error_distribution=show_error_distribution,
            show_error_examples=show_error_examples,
            max_error_examples=max_error_examples,
            variant=variant,
            comparison_run_binding=comparison_run_binding
        )

    def parse_diff_view(self, line: str) -> 'DiffView':
        """
        Parse diff view component for comparing text/code.
        
        Syntax: diff_view "view_id":
                  left_binding: "version.v1.output"
                  right_binding: "version.v2.output"
                  mode: "split"
                  content_type: "code"
                  language: "python"
        """
        from namel3ss.ast.pages import DiffView
        
        if not line.strip().endswith(':'):
            raise self._error(
                "Diff view declaration must end with ':'",
                self.pos,
                line,
                hint='Syntax: diff_view "view_id":'
            )
        
        # Extract view ID
        match = re.match(r'\s*diff_view\s+(?:"([^"]+)"|([a-zA-Z_][a-zA-Z0-9_]*))\s*:', line)
        if not match:
            raise self._error(
                'Diff view requires an ID',
                self.pos,
                line,
                hint='Syntax: diff_view "view_id":'
            )
        
        view_id = match.group(1) or match.group(2)
        base_indent = self._indent(line)
        
        # Default values
        left_binding = ""
        right_binding = ""
        mode = "split"
        content_type = "text"
        language = None
        ignore_whitespace = False
        ignore_case = False
        context_lines = 3
        show_line_numbers = True
        highlight_inline_changes = True
        show_legend = True
        max_height = None
        enable_copy = True
        enable_download = False
        
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
            
            self._advance()
            
            if stripped.startswith('left_binding:'):
                left_binding = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('right_binding:'):
                right_binding = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('mode:'):
                mode = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('content_type:'):
                content_type = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('language:'):
                language = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('ignore_whitespace:'):
                ignore_whitespace = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('ignore_case:'):
                ignore_case = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('context_lines:'):
                context_lines = int(stripped.split(':', 1)[1].strip())
            elif stripped.startswith('show_line_numbers:'):
                show_line_numbers = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('highlight_inline_changes:'):
                highlight_inline_changes = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('show_legend:'):
                show_legend = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('max_height:'):
                max_height = stripped.split(':', 1)[1].strip().strip('"')
            elif stripped.startswith('enable_copy:'):
                enable_copy = stripped.split(':', 1)[1].strip().lower() == 'true'
            elif stripped.startswith('enable_download:'):
                enable_download = stripped.split(':', 1)[1].strip().lower() == 'true'
        
        if not left_binding:
            raise self._error(
                "Diff view requires 'left_binding' property",
                self.pos - 1,
                line,
                hint="Example: left_binding: \"version.v1.output\""
            )
        
        if not right_binding:
            raise self._error(
                "Diff view requires 'right_binding' property",
                self.pos - 1,
                line,
                hint="Example: right_binding: \"version.v2.output\""
            )
        
        return DiffView(
            id=view_id,
            left_binding=left_binding,
            right_binding=right_binding,
            mode=mode,
            content_type=content_type,
            language=language,
            ignore_whitespace=ignore_whitespace,
            ignore_case=ignore_case,
            context_lines=context_lines,
            show_line_numbers=show_line_numbers,
            highlight_inline_changes=highlight_inline_changes,
            show_legend=show_legend,
            max_height=max_height,
            enable_copy=enable_copy,
            enable_download=enable_download
        )


