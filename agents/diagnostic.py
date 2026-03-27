"""
Diagnostic Agent — Smart Log Parsing
Key insight: Local Python parsing FIRST, LLM SECOND.
Never feed the full log to the LLM — AiiDA logs can be several MBs.
"""

import re
from dataclasses import dataclass
from aiida import load_profile
from aiida.orm import load_node, CalcJobNode

load_profile()

# Known exit codes and their meanings
# Sourced from aiida-quantumespresso exit code definitions
EXIT_CODE_MAP = {
    100: "Calculation did not reach a terminal state.",
    110: "Calculation finished but output parsing failed.",
    200: "Calculation failed with unrecoverable error in setup.",
    300: "Calculation failed with unrecoverable error during calculation.",
    310: "Calculation ran out of walltime.",
    320: "Calculation ran out of memory.",
    410: "SCF convergence not reached within the maximum number of iterations.",
    411: "SCF convergence not reached: bfgs history limit exceeded.",
    420: "Ionic convergence not reached within the maximum number of steps.",
    500: "Calculation failed with an unrecoverable error.",
}

# Suggested fixes per exit code
FIX_SUGGESTIONS = {
    410: [
        "Increase `electron_maxstep` from default (100) to 200 or higher.",
        "Reduce `conv_thr` from 1e-8 to 1e-6 for less strict convergence.",
        "Try a different `mixing_mode` (e.g. 'local-TF' instead of 'plain').",
        "Reduce `mixing_beta` to 0.3 or lower to stabilise SCF mixing.",
    ],
    411: [
        "Increase `bfgs_ndim` to allow a longer BFGS history.",
        "Switch to `ion_dynamics = 'damp'` for more stable ionic relaxation.",
    ],
    420: [
        "Increase `nstep` (max ionic steps) beyond the default.",
        "Tighten `forc_conv_thr` or loosen it depending on your target accuracy.",
    ],
    310: [
        "Request more walltime in your scheduler settings.",
        "Reduce system size or use k-point parallelisation to speed up the run.",
    ],
    320: [
        "Increase memory allocation per node in your scheduler settings.",
        "Reduce `ecutwfc` slightly to decrease memory footprint.",
    ],
}

# Keywords to search for in logs (no LLM needed for this step)
ERROR_KEYWORDS = [
    "convergence NOT achieved",
    "not converged",
    "ERROR",
    "Warning",
    "%%%",
    "stopping",
    "segmentation fault",
    "OOM",
    "out of memory",
    "walltime",
    "BFGS history",
]


@dataclass
class DiagnosticResult:
    pk: int
    exit_status: int
    exit_message: str
    extracted_snippets: list[str]
    diagnosis: str
    suggestions: list[str]


def extract_log_snippets(log_text: str, tail_lines: int = 80) -> list[str]:
    """
    Step 1: Local Python parsing — no LLM involved.
    Extract only relevant lines from the log using keyword matching
    and tail extraction. Keeps token usage minimal for the LLM step.
    """
    lines = log_text.splitlines()

    # Always grab the tail of the log
    tail = lines[-tail_lines:] if len(lines) > tail_lines else lines

    # Find keyword matches anywhere in the full log
    keyword_hits = []
    for i, line in enumerate(lines):
        if any(kw.lower() in line.lower() for kw in ERROR_KEYWORDS):
            # Include surrounding context (2 lines before and after)
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            snippet = "\n".join(lines[start:end])
            if snippet not in keyword_hits:
                keyword_hits.append(snippet)

    return keyword_hits + ["\n".join(tail)]


def diagnose_calculation(pk: int) -> DiagnosticResult:
    """
    Main diagnostic function.
    Step 1: Local Python parsing (exit code + keyword extraction)
    Step 2: Snippet preparation for LLM (max ~500 tokens)
    Step 3: ChromaDB cross-reference for fix suggestions (see search.py)
    """
    node = load_node(pk)

    if not isinstance(node, CalcJobNode):
        raise ValueError(f"Node {pk} is not a CalcJobNode")

    exit_status = node.exit_status or 0
    exit_message = node.exit_message or ""

    # Retrieve the retrieved folder and read stdout log
    try:
        retrieved = node.outputs.retrieved
        log_text = retrieved.base.repository.get_object_content("_scheduler-stdout.txt")
    except Exception:
        log_text = ""

    # Step 1: Extract snippets locally — no LLM
    snippets = extract_log_snippets(log_text) if log_text else []

    # Step 2: Build human-readable diagnosis from exit code map
    known_cause = EXIT_CODE_MAP.get(exit_status, "Unknown exit code.")
    suggestions = FIX_SUGGESTIONS.get(exit_status, [
        "Check the full log for clues.",
        "Consult the AiiDA documentation for this exit code.",
    ])

    diagnosis = (
        f"Your calculation (pk={pk}) failed with exit code {exit_status}.\n"
        f"Cause: {known_cause}\n"
        f"Exit message: {exit_message or '(none)'}"
    )

    return DiagnosticResult(
        pk=pk,
        exit_status=exit_status,
        exit_message=exit_message,
        extracted_snippets=snippets,
        diagnosis=diagnosis,
        suggestions=suggestions,
    )


def format_diagnostic_output(result: DiagnosticResult) -> str:
    """Format the diagnostic result for display to the user."""
    lines = [
        f"=== Diagnostic Report for pk={result.pk} ===",
        "",
        result.diagnosis,
        "",
        "Suggested fixes:",
    ]
    for i, suggestion in enumerate(result.suggestions, 1):
        lines.append(f"  {i}. {suggestion}")

    if result.extracted_snippets:
        lines += [
            "",
            "--- Relevant log snippets (extracted locally) ---",
            result.extracted_snippets[0][:800],  # Cap at ~500 tokens
        ]

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python diagnostic.py <pk>")
        sys.exit(1)

    pk = int(sys.argv[1])
    result = diagnose_calculation(pk)
    print(format_diagnostic_output(result))
