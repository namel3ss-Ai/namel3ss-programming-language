#!/bin/bash
# Namel3ss Debugging System Demo
# This script demonstrates the debugging capabilities we just implemented

echo "🚀 Namel3ss Debugging System Demo"
echo "=================================="
echo

echo "1️⃣  Setting up debugging environment..."
export NAMEL3SS_DEBUG_ENABLED=true
export NAMEL3SS_DEBUG_LEVEL=info
export NAMEL3SS_DEBUG_OUTPUT_DIR=./debug_traces
echo "✅ Debug environment configured"
echo

echo "2️⃣  Analyzing sample execution trace..."
echo "📊 Execution Summary:"
namel3ss debug analyze debug_traces/sample_trace.jsonl --summary
echo

echo "3️⃣  Inspecting specific events..."
echo "🔍 First event details:"
namel3ss debug inspect debug_traces/sample_trace.jsonl --event 0
echo

echo "4️⃣  Filtering events by agent..."
echo "🤖 ResearchAgent events:"
namel3ss debug inspect debug_traces/sample_trace.jsonl --agent ResearchAgent
echo

echo "5️⃣  Checking for errors..."
echo "🚨 Error analysis:"
namel3ss debug analyze debug_traces/sample_trace.jsonl --errors
echo

echo "6️⃣  Performance analysis..."
echo "⚡ Performance metrics:"
namel3ss debug analyze debug_traces/sample_trace.jsonl --performance
echo

echo "7️⃣  Replay functionality (non-interactive)..."
echo "🔄 Full trace replay:"
namel3ss debug replay debug_traces/sample_trace.jsonl
echo

echo "✅ Debugging system demo complete!"
echo
echo "🎯 Key Features Demonstrated:"
echo "   • Execution tracing and analysis"
echo "   • Event inspection and filtering"  
echo "   • Error detection and reporting"
echo "   • Performance analysis"
echo "   • Deterministic replay"
echo "   • Beautiful CLI interface with Rich formatting"
echo
echo "💡 To run interactive replay:"
echo "   namel3ss debug replay debug_traces/sample_trace.jsonl --step"
echo
echo "📚 For more information:"
echo "   namel3ss debug --help"