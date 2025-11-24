"""
CLI debug commands for namel3ss debugging and observability.

Provides 'namel3ss debug' command with subcommands for tracing, replay, analysis, and inspection.
"""

import asyncio
import json
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from namel3ss.debugging import (
    DebugConfiguration,
    TraceFilter,
    TraceEventType,
    initialize_tracing,
)
from namel3ss.debugging.tracer import ExecutionTracer
from namel3ss.debugging.replayer import ExecutionReplayer, ReplayBreakpoint, TraceAnalyzer
from namel3ss.debugging.config import load_debug_configuration
from namel3ss.cli.loading import load_n3_app

console = None

try:
    import rich.console
    import rich.table
    import rich.panel
    import rich.syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = rich.console.Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def _print(message: str, style: str = None):
    """Print with optional Rich styling."""
    if HAS_RICH and console and style:
        console.print(message, style=style)
    else:
        print(message)


def _print_table(headers: List[str], rows: List[List[str]], title: str = None):
    """Print a table with optional Rich formatting."""
    if HAS_RICH and console:
        table = rich.table.Table(title=title)
        for header in headers:
            table.add_column(header)
        for row in rows:
            table.add_row(*row)
        console.print(table)
    else:
        if title:
            print(f"\n{title}")
            print("=" * len(title))
        
        # Calculate column widths
        all_rows = [headers] + rows
        widths = [max(len(str(row[i])) for row in all_rows) for i in range(len(headers))]
        
        # Print headers
        header_row = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        print(header_row)
        print("-" * len(header_row))
        
        # Print rows
        for row in rows:
            print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))


@click.group()
def debug():
    """Debug namel3ss applications with execution tracing and replay."""
    pass


@debug.command()
@click.argument("app_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output trace file path")
@click.option("--filter", "-f", type=click.Choice(["agent", "prompt", "chain", "tool", "llm"]), 
              multiple=True, help="Filter events by component type")
@click.option("--format", type=click.Choice(["json", "jsonl"]), default="jsonl", 
              help="Trace file format")
@click.option("--memory/--no-memory", default=True, help="Capture memory usage")
@click.option("--performance/--no-performance", default=True, help="Capture performance metrics")
@click.option("--buffer-size", type=int, default=1000, help="Event buffer size")
@click.option("--run-args", type=str, help="JSON string of arguments to pass to app")
def trace(
    app_path: str,
    output: Optional[str],
    filter: tuple,
    format: str,
    memory: bool,
    performance: bool,
    buffer_size: int,
    run_args: Optional[str],
):
    """
    Trace execution of a namel3ss application.
    
    Records detailed execution events to a trace file for later replay and analysis.
    
    Examples:
        namel3ss debug trace my_app.n3
        namel3ss debug trace my_app.n3 --output trace.jsonl --filter agent --filter llm
        namel3ss debug trace my_app.n3 --run-args '{"input": "Hello world"}'
    """
    asyncio.run(_trace_app(
        app_path, output, filter, format, memory, performance, buffer_size, run_args
    ))


async def _trace_app(
    app_path: str,
    output: Optional[str],
    filter_components: tuple,
    format: str,
    memory: bool,
    performance: bool,
    buffer_size: int,
    run_args: Optional[str],
):
    """Execute app with tracing enabled."""
    
    # Parse run arguments
    args = {}
    if run_args:
        try:
            args = json.loads(run_args)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing run-args: {e}[/red]")
            return
    
    # Setup debug configuration
    config = DebugConfiguration(
        enabled=True,
        trace_output_dir=Path(output).parent if output else Path("./debug/traces"),
        buffer_size=buffer_size,
        capture_memory_usage=memory,
        capture_performance_markers=performance,
    )
    
    if output:
        # Custom output file
        config.auto_trace_filename = False
        
    # Setup filtering
    if filter_components:
        config.trace_filter = TraceFilter(
            components=set(filter_components)
        )
    
    # Initialize tracer
    tracer = initialize_tracing(config)
    
    console.print(f"[green]Starting trace of {app_path}[/green]")
    
    try:
        # Start execution trace
        context = await tracer.start_execution_trace(
            app_name=Path(app_path).stem,
            execution_id=f"debug_trace_{int(asyncio.get_event_loop().time())}"
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Tracing execution...", total=None)
            
            # Load and run the app
            try:
                app = load_n3_app(app_path)
                
                # TODO: Integrate with actual app execution
                # For now, simulate execution
                await asyncio.sleep(1)  # Placeholder for actual execution
                
                progress.update(task, description="Execution complete")
                
            except Exception as e:
                console.print(f"[red]Error during execution: {e}[/red]")
                return
        
        # End trace and write file
        trace_file = await tracer.end_execution_trace()
        
        if trace_file:
            console.print(f"[green]Trace written to: {trace_file}[/green]")
            
            # Show basic stats
            replayer = ExecutionReplayer(trace_file)
            summary = replayer.get_execution_summary()
            
            table = rich.table.Table(title="Execution Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            exec_overview = summary["execution_overview"]
            table.add_row("Total Events", str(exec_overview["total_events"]))
            table.add_row("Duration", f"{exec_overview['execution_duration_seconds']:.2f}s")
            table.add_row("Error Count", str(exec_overview["error_count"]))
            table.add_row("Success Rate", f"{exec_overview['success_rate']:.1%}")
            
            console.print(table)
        else:
            console.print("[yellow]Tracing was disabled or failed[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Tracing failed: {e}[/red]")


@debug.command()
@click.argument("trace_file", type=click.Path(exists=True))
@click.option("--step/--no-step", default=False, help="Step through events one by one")
@click.option("--breakpoint", "-b", multiple=True, 
              help="Set breakpoint (format: event_type:index or component:name)")
@click.option("--mock", "-m", multiple=True,
              help="Mock responses (format: component_name:response_json)")
@click.option("--filter", "-f", type=click.Choice(["agent", "prompt", "chain", "tool", "llm"]),
              multiple=True, help="Filter events by component type")
def replay(
    trace_file: str,
    step: bool,
    breakpoint: tuple,
    mock: tuple,
    filter: tuple,
):
    """
    Replay execution from a trace file.
    
    Provides deterministic replay with debugging capabilities including breakpoints
    and mock response injection.
    
    Examples:
        namel3ss debug replay trace.jsonl
        namel3ss debug replay trace.jsonl --step
        namel3ss debug replay trace.jsonl --breakpoint agent_turn_start:3
        namel3ss debug replay trace.jsonl --mock "MyAgent:{'response': 'mocked'}"
    """
    asyncio.run(_replay_trace(trace_file, step, breakpoint, mock, filter))


async def _replay_trace(
    trace_file: str,
    step: bool,
    breakpoint_specs: tuple,
    mock_specs: tuple,
    filter_components: tuple,
):
    """Execute trace replay."""
    
    # Parse breakpoints
    breakpoints = []
    for bp_spec in breakpoint_specs:
        try:
            if ":" in bp_spec:
                key, value = bp_spec.split(":", 1)
                if key in [e.value for e in TraceEventType]:
                    # Event type breakpoint
                    breakpoints.append(ReplayBreakpoint(
                        event_type=TraceEventType(key),
                        event_index=int(value) if value.isdigit() else None
                    ))
                else:
                    # Component breakpoint
                    breakpoints.append(ReplayBreakpoint(
                        component=key,
                        component_name=value
                    ))
            else:
                console.print(f"[yellow]Invalid breakpoint format: {bp_spec}[/yellow]")
        except Exception as e:
            console.print(f"[red]Error parsing breakpoint {bp_spec}: {e}[/red]")
    
    # Parse mock responses
    mock_responses = {}
    for mock_spec in mock_specs:
        try:
            if ":" in mock_spec:
                component, response_json = mock_spec.split(":", 1)
                mock_responses[component] = json.loads(response_json)
        except Exception as e:
            console.print(f"[red]Error parsing mock {mock_spec}: {e}[/red]")
    
    # Setup filtering
    trace_filter = None
    if filter_components:
        trace_filter = TraceFilter(
            components=set(filter_components)
        )
    
    # Create replayer
    try:
        replayer = ExecutionReplayer(
            Path(trace_file),
            filter=trace_filter,
            breakpoints=breakpoints,
            mock_responses=mock_responses,
        )
    except Exception as e:
        console.print(f"[red]Failed to load trace file: {e}[/red]")
        return
    
    console.print(f"[green]Loaded trace with {len(replayer.events)} events[/green]")
    
    if step:
        await _interactive_replay(replayer)
    else:
        # Full replay
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Replaying trace...", total=len(replayer.events))
            
            state = replayer.replay_full()
            progress.update(task, completed=len(replayer.events))
        
        console.print(f"[green]Replay completed - {state.total_events} events processed[/green]")


async def _interactive_replay(replayer: ExecutionReplayer):
    """Interactive step-by-step replay."""
    console.print("[cyan]Interactive replay mode - Press Enter to step, 'q' to quit[/cyan]")
    
    while not replayer.state.completed:
        # Show current state
        progress = replayer.state.get_progress_percentage()
        console.print(f"\n[dim]Progress: {progress:.1f}% ({replayer.state.current_event_index}/{replayer.state.total_events})[/dim]")
        
        # Step to next event
        event = replayer.replay_step()
        if not event:
            break
        
        # Display event
        _display_event(event)
        
        # Check for pause
        if replayer.state.paused:
            console.print("[yellow]Hit breakpoint - paused[/yellow]")
        
        # Wait for user input
        user_input = input("\n[Enter] to continue, [q] to quit: ").strip().lower()
        if user_input == 'q':
            break
    
    console.print("[green]Replay completed[/green]")


def _display_event(event):
    """Display a trace event in a formatted panel."""
    
    # Create event summary
    title = f"{event.event_type.value} - {event.component}"
    if event.component_name:
        title += f"/{event.component_name}"
    
    content = []
    
    # Status and timing
    status_style = "green" if event.status == "completed" else "red" if event.status == "failed" else "yellow"
    content.append(f"Status: [{status_style}]{event.status}[/{status_style}]")
    
    if event.duration_ms:
        content.append(f"Duration: {event.duration_ms:.2f}ms")
    
    # Error if present
    if event.error:
        content.append(f"Error: [red]{event.error}[/red]")
    
    # Inputs/outputs (truncated)
    if event.inputs:
        inputs_preview = str(event.inputs)[:100]
        if len(str(event.inputs)) > 100:
            inputs_preview += "..."
        content.append(f"Inputs: {inputs_preview}")
    
    if event.outputs:
        outputs_preview = str(event.outputs)[:100]
        if len(str(event.outputs)) > 100:
            outputs_preview += "..."
        content.append(f"Outputs: {outputs_preview}")
    
    panel = rich.panel.Panel(
        "\n".join(content),
        title=title,
        border_style="blue",
    )
    
    console.print(panel)


@debug.command()
@click.argument("trace_file", type=click.Path(exists=True))
@click.option("--performance/--no-performance", default=False, help="Show performance analysis")
@click.option("--errors/--no-errors", default=False, help="Show error analysis")
@click.option("--summary/--no-summary", default=True, help="Show execution summary")
@click.option("--format", type=click.Choice(["table", "json"]), default="table", help="Output format")
def analyze(
    trace_file: str,
    performance: bool,
    errors: bool,
    summary: bool,
    format: str,
):
    """
    Analyze a trace file for insights and patterns.
    
    Provides detailed analysis of execution performance, errors, and statistics.
    
    Examples:
        namel3ss debug analyze trace.jsonl
        namel3ss debug analyze trace.jsonl --performance --errors
        namel3ss debug analyze trace.jsonl --format json
    """
    try:
        analyzer = TraceAnalyzer(Path(trace_file))
    except Exception as e:
        console.print(f"[red]Failed to load trace: {e}[/red]")
        return
    
    results = {}
    
    # Execution summary
    if summary:
        results["summary"] = analyzer.replayer.get_execution_summary()
    
    # Performance analysis
    if performance:
        results["performance"] = analyzer.analyze_performance()
    
    # Error analysis
    if errors:
        results["errors"] = analyzer.analyze_errors()
    
    # Output results
    if format == "json":
        console.print(json.dumps(results, indent=2))
    else:
        _display_analysis_results(results)


def _display_analysis_results(results: Dict[str, Any]):
    """Display analysis results in formatted tables."""
    
    if "summary" in results:
        summary = results["summary"]["execution_overview"]
        
        table = rich.table.Table(title="Execution Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Events", str(summary["total_events"]))
        table.add_row("Duration", f"{summary['execution_duration_seconds']:.2f}s")
        table.add_row("Error Count", str(summary["error_count"]))
        table.add_row("Success Rate", f"{summary['success_rate']:.1%}")
        
        console.print(table)
    
    if "performance" in results:
        perf = results["performance"]
        
        # LLM performance
        if "llm_performance" in perf:
            llm_perf = perf["llm_performance"]
            table = rich.table.Table(title="LLM Performance")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Calls", str(llm_perf["total_calls"]))
            table.add_row("Avg Duration", f"{llm_perf['avg_duration_ms']:.2f}ms")
            table.add_row("Max Duration", f"{llm_perf['max_duration_ms']:.2f}ms")
            table.add_row("Total Tokens", str(llm_perf["total_tokens"]))
            
            console.print(table)
        
        # Slow operations
        if "slow_operations" in perf and perf["slow_operations"]:
            table = rich.table.Table(title="Slow Operations (>1s)")
            table.add_column("Component", style="cyan")
            table.add_column("Name", style="yellow")
            table.add_column("Duration", style="red")
            
            for op in perf["slow_operations"][:10]:  # Show top 10
                table.add_row(
                    op["component"],
                    op.get("component_name", ""),
                    f"{op['duration_ms']:.2f}ms"
                )
            
            console.print(table)
    
    if "errors" in results:
        errors = results["errors"]
        
        table = rich.table.Table(title="Error Analysis")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="red")
        table.add_column("Examples", style="yellow")
        
        for category, data in errors["error_categories"].items():
            examples = ", ".join([
                f"{ex['component']}/{ex.get('component_name', '')}"
                for ex in data["examples"][:3]
            ])
            table.add_row(category, str(data["count"]), examples)
        
        console.print(table)


@debug.command()
@click.argument("trace_file", type=click.Path(exists=True))
@click.option("--event", "-e", type=int, help="Inspect specific event by index")
@click.option("--agent", "-a", help="Show events for specific agent")
@click.option("--chain", "-c", help="Show events for specific chain")
@click.option("--prompt", "-p", help="Show events for specific prompt")
@click.option("--errors-only", is_flag=True, help="Show only error events")
def inspect(
    trace_file: str,
    event: Optional[int],
    agent: Optional[str],
    chain: Optional[str], 
    prompt: Optional[str],
    errors_only: bool,
):
    """
    Inspect specific events or components in a trace file.
    
    Provides detailed examination of individual events or filtered views.
    
    Examples:
        namel3ss debug inspect trace.jsonl --event 42
        namel3ss debug inspect trace.jsonl --agent MyAgent
        namel3ss debug inspect trace.jsonl --errors-only
    """
    try:
        replayer = ExecutionReplayer(Path(trace_file))
    except Exception as e:
        console.print(f"[red]Failed to load trace: {e}[/red]")
        return
    
    events = replayer.events
    
    if event is not None:
        # Inspect specific event
        if 0 <= event < len(events):
            _display_detailed_event(events[event])
        else:
            console.print(f"[red]Event index {event} out of range (0-{len(events)-1})[/red]")
    
    elif agent:
        # Show agent events
        agent_events = [e for e in events if e.component == "agent" and e.component_name == agent]
        _display_event_list(agent_events, f"Agent: {agent}")
    
    elif chain:
        # Show chain events
        chain_events = [e for e in events if e.component == "chain" and e.component_name == chain]
        _display_event_list(chain_events, f"Chain: {chain}")
    
    elif prompt:
        # Show prompt events
        prompt_events = [e for e in events if e.component == "prompt" and e.component_name == prompt]
        _display_event_list(prompt_events, f"Prompt: {prompt}")
    
    elif errors_only:
        # Show error events
        error_events = replayer.get_error_events()
        _display_event_list(error_events, "Error Events")
    
    else:
        # Show overall summary
        table = rich.table.Table(title="Trace Overview")
        table.add_column("Index", style="dim")
        table.add_column("Event Type", style="cyan")
        table.add_column("Component", style="green")
        table.add_column("Name", style="yellow")
        table.add_column("Status", style="white")
        table.add_column("Duration", style="blue")
        
        for i, e in enumerate(events[:20]):  # Show first 20 events
            status_style = "green" if e.status == "completed" else "red" if e.status == "failed" else "yellow"
            duration = f"{e.duration_ms:.2f}ms" if e.duration_ms else ""
            
            table.add_row(
                str(i),
                e.event_type.value,
                e.component,
                e.component_name or "",
                f"[{status_style}]{e.status}[/{status_style}]",
                duration
            )
        
        console.print(table)
        
        if len(events) > 20:
            console.print(f"[dim]... and {len(events) - 20} more events[/dim]")


def _display_detailed_event(event):
    """Display detailed view of a single event."""
    
    # Event header
    title = f"Event: {event.event_type.value}"
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    
    # Basic info table
    table = rich.table.Table(show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("ID", event.event_id)
    table.add_row("Component", f"{event.component}/{event.component_name or 'N/A'}")
    table.add_row("Status", event.status)
    table.add_row("Timestamp", str(event.timestamp))
    
    if event.duration_ms:
        table.add_row("Duration", f"{event.duration_ms:.2f}ms")
    
    if event.parent_event_id:
        table.add_row("Parent ID", event.parent_event_id)
    
    if event.tokens_used:
        table.add_row("Tokens", str(event.tokens_used))
    
    if event.memory_usage_mb:
        table.add_row("Memory", f"{event.memory_usage_mb:.1f}MB")
    
    if event.error:
        table.add_row("Error", f"[red]{event.error}[/red]")
    
    console.print(table)
    
    # Inputs
    if event.inputs:
        console.print("\n[bold]Inputs:[/bold]")
        syntax = rich.syntax.Syntax(
            json.dumps(event.inputs, indent=2),
            "json",
            theme="monokai",
            line_numbers=True
        )
        console.print(syntax)
    
    # Outputs
    if event.outputs:
        console.print("\n[bold]Outputs:[/bold]")
        syntax = rich.syntax.Syntax(
            json.dumps(event.outputs, indent=2),
            "json",
            theme="monokai",
            line_numbers=True
        )
        console.print(syntax)
    
    # Metadata
    if event.metadata:
        console.print("\n[bold]Metadata:[/bold]")
        syntax = rich.syntax.Syntax(
            json.dumps(event.metadata, indent=2),
            "json",
            theme="monokai",
            line_numbers=True
        )
        console.print(syntax)


def _display_event_list(events: List, title: str):
    """Display a list of events in table format."""
    
    if not events:
        console.print(f"[yellow]No events found for {title}[/yellow]")
        return
    
    table = rich.table.Table(title=f"{title} ({len(events)} events)")
    table.add_column("Index", style="dim")
    table.add_column("Event Type", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Duration", style="blue")
    table.add_column("Timestamp", style="dim")
    
    for i, event in enumerate(events):
        status_style = "green" if event.status == "completed" else "red" if event.status == "failed" else "yellow"
        duration = f"{event.duration_ms:.2f}ms" if event.duration_ms else ""
        
        table.add_row(
            str(i),
            event.event_type.value,
            f"[{status_style}]{event.status}[/{status_style}]",
            duration,
            f"{event.timestamp:.3f}"
        )
    
    console.print(table)


if __name__ == "__main__":
    debug()