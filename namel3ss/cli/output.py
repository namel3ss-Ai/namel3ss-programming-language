"""
Output formatting for CLI operations.

This module provides functions for pretty-printing various response types
including chain predictions, experiment results, and backend summaries.
"""

import json
from typing import Any, Dict


def print_prediction_response(response: Dict[str, Any]) -> None:
    """
    Pretty-print chain prediction output with status-aware messaging.
    
    Handles different response statuses (success, error, partial) and
    formats output appropriately with model info, inputs, results, and metadata.
    
    Args:
        response: Prediction response dictionary from chain execution
    
    Examples:
        >>> response = {
        ...     "status": "success",
        ...     "model": "gpt-4",
        ...     "framework": "openai",
        ...     "result": {"answer": "42"}
        ... }
        >>> print_prediction_response(response)  # doctest: +SKIP
        Model: gpt-4 (framework: openai, version: n/a)
        Result:
        {
          "answer": "42"
        }
    """
    status = str(response.get("status") or response.get("output", {}).get("status") or "").lower()
    framework = response.get("framework") or "n/a"
    version = response.get("version") or "n/a"
    
    # Print model info
    print(f"Model: {response.get('model', 'n/a')} (framework: {framework}, version: {version})")
    
    # Handle error status
    if status == "error":
        error_code = response.get("error") or response.get("output", {}).get("error") or "unknown_error"
        detail = response.get("detail") or response.get("output", {}).get("detail")
        print(f"[error] {error_code}")
        if detail:
            print(f"Detail: {detail}")
        return
    
    # Handle partial status
    if status == "partial":
        print("[warning] Partial result reported by downstream components.")
    
    # Print inputs
    inputs = response.get("inputs") or response.get("input")
    if inputs:
        print("Inputs:")
        print(json.dumps(inputs, indent=2))
    
    # Print result or output
    if "result" in response:
        print("Result:")
        result = response["result"]
        if isinstance(result, dict):
            print(json.dumps(result, indent=2))
        else:
            print(result)
    elif "output" in response:
        print("Output:")
        print(json.dumps(response["output"], indent=2))
    
    # Print notes if present
    if response.get("notes"):
        print("Notes:")
        print(json.dumps(response["notes"], indent=2))
    
    # Print metadata
    metadata = response.get("metadata", {})
    if metadata:
        print("Metadata:")
        print(json.dumps(metadata, indent=2))


def print_experiment_result(result: Dict[str, Any]) -> None:
    """
    Pretty-print experiment evaluation output with status awareness.
    
    Formats experiment results including variant comparisons, metrics,
    and winner determination with proper handling of error states.
    
    Args:
        result: Experiment result dictionary from evaluation
    
    Examples:
        >>> result = {
        ...     "experiment": "prompt_comparison",
        ...     "status": "success",
        ...     "winner": "variant_a",
        ...     "variants": [
        ...         {"name": "variant_a", "score": 0.95, "target_type": "prompt", "target_name": "v1"},
        ...         {"name": "variant_b", "score": 0.87, "target_type": "prompt", "target_name": "v2"}
        ...     ]
        ... }
        >>> print_experiment_result(result)  # doctest: +SKIP
        Experiment: prompt_comparison
        Status: success
        Winner: variant_a
        Variants:
          - variant_a (prompt:v1) score=0.950
          - variant_b (prompt:v2) score=0.870
    """
    status = str(result.get("status") or "").lower()
    
    # Print header
    print(f"Experiment: {result.get('experiment', 'n/a')}")
    print(f"Status: {status or 'unknown'}")
    
    # Handle error status
    if status == "error":
        error_code = result.get("error", "experiment_error")
        detail = result.get("detail")
        print(f"[error] {error_code}")
        if detail:
            print(f"Detail: {detail}")
        return
    
    # Handle partial status
    if status == "partial":
        print("[warning] Partial metrics returned; inspect notes for details.")
    
    # Print winner
    print(f"Winner: {result.get('winner') or 'n/a'}")
    
    # Print variants
    variants = result.get("variants") or []
    if variants:
        print("Variants:")
        for entry in variants:
            target = f"{entry.get('target_type')}:{entry.get('target_name') or 'default'}"
            score = entry.get("score")
            score_display = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
            print(f"  - {entry.get('name')} ({target}) score={score_display}")
    
    # Print metrics
    metrics = result.get("metrics") or []
    if metrics:
        print("Metrics:")
        for metric in metrics:
            name = metric.get("name", "metric")
            value = metric.get("value")
            value_display = f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
            print(f"  - {name}: {value_display}")
    
    # Print inputs if present
    if result.get("inputs"):
        print("Inputs:")
        print(json.dumps(result["inputs"], indent=2))
    
    # Print notes if present
    if result.get("notes"):
        print("Notes:")
        print(json.dumps(result["notes"], indent=2))


def print_success(message: str) -> None:
    """
    Print success message with checkmark prefix.
    
    Args:
        message: Success message to display
    
    Examples:
        >>> print_success("Build completed successfully")
        ✓ Build completed successfully
    """
    print(f"✓ {message}")


def print_error(message: str) -> None:
    """
    Print error message with cross prefix.
    
    Args:
        message: Error message to display
    
    Examples:
        >>> print_error("Build failed")
        ✗ Build failed
    """
    print(f"✗ {message}")


def print_warning(message: str) -> None:
    """
    Print warning message with warning prefix.
    
    Args:
        message: Warning message to display
    
    Examples:
        >>> print_warning("Deprecated feature used")
        ⚠ Deprecated feature used
    """
    print(f"⚠ {message}")


def print_info(message: str) -> None:
    """
    Print informational message with info prefix.
    
    Args:
        message: Informational message to display
    
    Examples:
        >>> print_info("Loading configuration...")
        ℹ Loading configuration...
    """
    print(f"ℹ {message}")
