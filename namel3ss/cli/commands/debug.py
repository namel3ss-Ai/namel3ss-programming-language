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
from namel3ss.debugging.config import get_debug_config_manager
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


def cmd_debug(args: argparse.Namespace) -> int:
    """Main debug command dispatcher."""
    if not hasattr(args, 'debug_action') or args.debug_action is None:
        print("usage: namel3ss debug {trace,replay,analyze,inspect} ...")
        print("\nDebug namel3ss applications with execution tracing and replay.")
        print("\nSubcommands:")
        print("  trace     Record execution trace of namel3ss application")
        print("  replay    Replay execution trace with step-by-step debugging")
        print("  analyze   Analyze execution trace for performance and errors")
        print("  inspect   Inspect specific events in execution trace")
        return 1
    
    # Dispatch to appropriate subcommand
    if args.debug_action == 'trace':
        return _cmd_debug_trace(args)
    elif args.debug_action == 'replay':
        return _cmd_debug_replay(args)
    elif args.debug_action == 'analyze':
        return _cmd_debug_analyze(args)
    elif args.debug_action == 'inspect':
        return _cmd_debug_inspect(args)
    else:
        print(f"Unknown debug action: {args.debug_action}")
        return 1


async def _trace_app_execution(app_path: Path, output_file: Path, event_filter: Optional[List[str]] = None) -> int:
    """Execute app with tracing enabled."""
    try:
        # Initialize debugging
        config_manager = get_debug_config_manager()
        config = config_manager.get_runtime_config()
        config.enabled = True
        config.trace_file = str(output_file)
        
        if event_filter:
            # Convert filter strings to event types
            filter_types = []
            for f in event_filter:
                if f == "agent":
                    filter_types.extend([TraceEventType.AGENT_EXECUTION_START, TraceEventType.AGENT_EXECUTION_END])
                elif f == "prompt":
                    filter_types.extend([TraceEventType.PROMPT_EXECUTION_START, TraceEventType.PROMPT_EXECUTION_END])
                elif f == "chain":
                    filter_types.extend([TraceEventType.CHAIN_EXECUTION_START, TraceEventType.CHAIN_EXECUTION_END])
                elif f == "tool":
                    filter_types.extend([TraceEventType.TOOL_CALL_START, TraceEventType.TOOL_CALL_END])
                elif f == "llm":
                    filter_types.extend([TraceEventType.LLM_CALL_START, TraceEventType.LLM_CALL_END])
            
            config.event_filter = TraceFilter(event_types=filter_types)
        
        initialize_tracing(config)
        
        _print(f"🔍 Starting trace of {app_path}", "blue")
        _print(f"📝 Trace output: {output_file}", "dim")
        
        # Load and execute the app
        app = load_n3_app(app_path)
        
        # TODO: Execute the app based on its type
        # This would need to integrate with the actual execution engine
        _print("⚠️  App execution integration not yet complete", "yellow")
        _print("📋 Trace file structure created", "green")
        
        return 0
        
    except Exception as e:
        _print(f"❌ Error during tracing: {e}", "red")
        return 1


def _cmd_debug_trace(args: argparse.Namespace) -> int:
    """Record execution trace of namel3ss application."""
    app_path = Path(args.app_path)
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f"{app_path.stem}_trace.jsonl")
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Run async tracing
    try:
        return asyncio.run(_trace_app_execution(app_path, output_file, args.filter))
    except KeyboardInterrupt:
        _print("\n⏹️  Tracing interrupted", "yellow")
        return 130  # Standard exit code for SIGINT


def _cmd_debug_replay(args: argparse.Namespace) -> int:
    """Replay execution trace with step-by-step debugging."""
    trace_file = Path(args.trace_file)
    
    if not trace_file.exists():
        _print(f"❌ Trace file not found: {trace_file}", "red")
        return 1
    
    try:
        replayer = ExecutionReplayer(str(trace_file))
        
        # Set up breakpoints if specified
        if args.breakpoint:
            for bp_spec in args.breakpoint:
                if ":" in bp_spec:
                    event_type, index = bp_spec.split(":", 1)
                    try:
                        event_type_enum = TraceEventType(event_type)
                        breakpoint = ReplayBreakpoint(
                            event_type=event_type_enum,
                            event_index=int(index) if index.isdigit() else None
                        )
                        replayer.add_breakpoint(breakpoint)
                    except ValueError:
                        _print(f"⚠️  Invalid breakpoint: {bp_spec}", "yellow")
        
        _print(f"🔄 Starting replay of {trace_file}", "blue")
        
        if args.step:
            # Interactive step-by-step replay
            _print("🎯 Interactive mode - press Enter to step, 'q' to quit", "dim")
            
            while True:
                current_event = replayer.state.current_event
                if replayer.state.completed:
                    _print("✅ Replay complete", "green")
                    break
                
                # Get the next event to be replayed
                if replayer.state.current_event_index < len(replayer.events):
                    next_event = replayer.events[replayer.state.current_event_index]
                    
                    # Show current event
                    _print(f"\n📍 Event {replayer.state.current_event_index}: {next_event.event_type.value}", "cyan")
                    if hasattr(next_event, 'metadata') and next_event.metadata:
                        _print(f"   Data: {json.dumps(next_event.metadata, indent=2)[:200]}...", "dim")
                else:
                    _print("✅ Replay complete", "green")
                    break
                
                # Wait for user input
                try:
                    user_input = input(">>> ").strip().lower()
                    if user_input in ['q', 'quit', 'exit']:
                        break
                except (EOFError, KeyboardInterrupt):
                    break
                
                # Step forward
                replayer.replay_step()
        else:
            # Non-interactive replay
            replayer.replay_full()
            _print(f"✅ Replayed {len(replayer.events)} events", "green")
        
        return 0
        
    except Exception as e:
        _print(f"❌ Error during replay: {e}", "red")
        return 1


def _cmd_debug_analyze(args: argparse.Namespace) -> int:
    """Analyze execution trace for performance and errors."""
    trace_file = Path(args.trace_file)
    
    if not trace_file.exists():
        _print(f"❌ Trace file not found: {trace_file}", "red")
        return 1
    
    try:
        analyzer = TraceAnalyzer(str(trace_file))
        
        if args.summary:
            # Show overall execution summary
            _print("📊 Execution Summary", "blue bold")
            _print("=" * 50)
            
            events = analyzer.replayer.events
            total_events = len(events)
            _print(f"Total Events: {total_events}")
            
            # Event type breakdown
            event_counts = {}
            for event in events:
                event_type = event.event_type.value
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            _print("\nEvent Type Breakdown:")
            for event_type, count in sorted(event_counts.items()):
                _print(f"  {event_type}: {count}")
        
        if args.performance:
            # Show performance analysis
            _print("\n⚡ Performance Analysis", "blue bold")
            _print("=" * 50)
            
            perf_analysis = analyzer.analyze_performance()
            _print("Performance analysis completed")
        
        if args.errors:
            # Show error analysis
            _print("\n🚨 Error Analysis", "blue bold") 
            _print("=" * 50)
            
            events = analyzer.replayer.events
            error_events = [e for e in events if e.event_type == TraceEventType.ERROR_OCCURRED]
            
            if not error_events:
                _print("✅ No errors found in trace", "green")
            else:
                for i, error_event in enumerate(error_events):
                    _print(f"Error {i + 1}: {error_event.metadata.get('error_type', 'Unknown')}")
                    if 'error_message' in error_event.metadata:
                        _print(f"  Message: {error_event.metadata['error_message']}")
                    _print(f"  Timestamp: {error_event.timestamp}")
        
        return 0
        
    except Exception as e:
        _print(f"❌ Error during analysis: {e}", "red")
        return 1


def _cmd_debug_inspect(args: argparse.Namespace) -> int:
    """Inspect specific events in execution trace."""
    trace_file = Path(args.trace_file)
    
    if not trace_file.exists():
        _print(f"❌ Trace file not found: {trace_file}", "red")
        return 1
    
    try:
        analyzer = TraceAnalyzer(str(trace_file))
        events = analyzer.replayer.events
        
        if args.event is not None:
            # Show specific event by index
            if 0 <= args.event < len(events):
                event = events[args.event]
                _print(f"🔍 Event {args.event}: {event.event_type.value}", "cyan bold")
                _print("=" * 50)
                _print(f"Timestamp: {event.timestamp}")
                _print(f"Status: {getattr(event, 'status', 'unknown')}")
                if hasattr(event, 'duration_ms') and event.duration_ms:
                    _print(f"Duration: {event.duration_ms:.2f}ms")
                
                if hasattr(event, 'metadata') and event.metadata:
                    _print("\nMetadata:")
                    _print(json.dumps(event.metadata, indent=2))
            else:
                _print(f"❌ Event index {args.event} out of range (0-{len(events)-1})", "red")
                return 1
        
        elif args.agent:
            # Show events for specific agent
            agent_events = [e for e in events if e.metadata and e.metadata.get('agent_name') == args.agent]
            if not agent_events:
                _print(f"No events found for agent: {args.agent}", "yellow")
            else:
                _print(f"🤖 Events for agent '{args.agent}' ({len(agent_events)} events)", "blue bold")
                _show_event_list(agent_events)
        
        elif args.chain:
            # Show events for specific chain
            chain_events = [e for e in events if e.metadata and e.metadata.get('chain_name') == args.chain]
            if not chain_events:
                _print(f"No events found for chain: {args.chain}", "yellow")
            else:
                _print(f"⛓️  Events for chain '{args.chain}' ({len(chain_events)} events)", "blue bold")
                _show_event_list(chain_events)
        
        else:
            # Show overview of all events
            _print(f"📋 All Events ({len(events)} total)", "blue bold")
            _show_event_list(events)
        
        return 0
        
    except Exception as e:
        _print(f"❌ Error during inspection: {e}", "red")
        return 1


def _show_event_list(events: List[Any]):
    """Display a list of events in a table format."""
    if not events:
        _print("No events to display", "dim")
        return
    
    headers = ["Index", "Event Type", "Status", "Duration", "Timestamp"]
    rows = []
    
    for i, event in enumerate(events):
        status = getattr(event, 'status', 'unknown')
        duration = f"{event.duration_ms:.2f}ms" if hasattr(event, 'duration_ms') and event.duration_ms else ""
        
        rows.append([
            str(i),
            event.event_type.value,
            status,
            duration,
            f"{event.timestamp:.3f}"
        ])
    
    _print_table(headers, rows)


def add_debug_command(subparsers) -> None:
    """Add debug command to the CLI parser."""
    debug_parser = subparsers.add_parser(
        'debug',
        help='Debug namel3ss applications with execution tracing and replay'
    )
    
    # Create subparser for debug actions
    debug_subparsers = debug_parser.add_subparsers(
        dest='debug_action',
        help='Debug actions'
    )
    
    # Trace command
    trace_parser = debug_subparsers.add_parser(
        'trace',
        help='Record execution trace of namel3ss application'
    )
    trace_parser.add_argument(
        'app_path',
        help='Path to namel3ss application file (.n3)'
    )
    trace_parser.add_argument(
        '--output', '-o',
        help='Output trace file path (defaults to <app>_trace.jsonl)'
    )
    trace_parser.add_argument(
        '--filter', '-f',
        choices=['agent', 'prompt', 'chain', 'tool', 'llm'],
        action='append',
        help='Filter events by component type (can be used multiple times)'
    )
    
    # Replay command
    replay_parser = debug_subparsers.add_parser(
        'replay',
        help='Replay execution trace with step-by-step debugging'
    )
    replay_parser.add_argument(
        'trace_file',
        help='Path to trace file (.jsonl)'
    )
    replay_parser.add_argument(
        '--step', '-s',
        action='store_true',
        help='Enable interactive step-by-step mode'
    )
    replay_parser.add_argument(
        '--breakpoint', '-b',
        action='append',
        help='Set breakpoint (format: event_type:index, e.g., agent_execution_start:0)'
    )
    
    # Analyze command
    analyze_parser = debug_subparsers.add_parser(
        'analyze',
        help='Analyze execution trace for performance and errors'
    )
    analyze_parser.add_argument(
        'trace_file',
        help='Path to trace file (.jsonl)'
    )
    analyze_parser.add_argument(
        '--performance', '-p',
        action='store_true',
        help='Show performance analysis'
    )
    analyze_parser.add_argument(
        '--errors', '-e',
        action='store_true', 
        help='Show error analysis'
    )
    analyze_parser.add_argument(
        '--summary', '-s',
        action='store_true',
        help='Show execution summary'
    )
    
    # Inspect command
    inspect_parser = debug_subparsers.add_parser(
        'inspect',
        help='Inspect specific events in execution trace'
    )
    inspect_parser.add_argument(
        'trace_file',
        help='Path to trace file (.jsonl)'
    )
    inspect_parser.add_argument(
        '--event', '-e',
        type=int,
        help='Inspect specific event by index'
    )
    inspect_parser.add_argument(
        '--agent', '-a',
        help='Show events for specific agent'
    )
    inspect_parser.add_argument(
        '--chain', '-c',
        help='Show events for specific chain'
    )
    
    debug_parser.set_defaults(func=cmd_debug)


__all__ = ['cmd_debug', 'add_debug_command']