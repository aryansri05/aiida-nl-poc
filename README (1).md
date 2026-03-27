# AiiDA Natural Language Interface — Proof of Concept

A minimal proof of concept for the [GSoC 2026 proposal](https://github.com/aryansri05/Resume): Natural Language Interface for AiiDA using Multi-Agent AI.

> **Status:** Early PoC — demonstrates core architecture. Full implementation planned for GSoC 2026.

---

## What This Demonstrates

This PoC shows the full architecture in miniature:

| Component | File | What it proves |
|---|---|---|
| FastMCP Server | `mcp_server/server.py` | AiiDA API exposed as typed, testable tools |
| Diagnostic Agent | `agents/diagnostic.py` | Smart log parsing — local Python first, LLM second |
| ChromaDB RAG | `agents/search.py` | Semantic search over error patterns and docs |
| Ollama Agent | `llm/ollama_agent.py` | Llama 3 tool calling, fully local, no API keys |

---

## Core Architecture

```
User (natural language)
        ↓
  Ollama / Llama 3
        ↓
   Tool Selection
    ↙         ↘
MCP Tools    ChromaDB
(AiiDA API)  (RAG search)
    ↘         ↙
   Final Response
```

**Key design decision:** The LLM never touches AiiDA directly. It only calls typed MCP tools. This makes the system safe, testable, and predictable.

---

## Smart Log Parsing (Diagnostic Agent)

AiiDA logs can be several MBs. Feeding the full log to an LLM would exceed the context window and produce unreliable results. Instead:

```
Step 1 → Local Python parsing (NO LLM):
          - Keyword search: "ERROR", "not converged", "Warning"
          - Extract log tail (last 80 lines)
          - Read exit_status from CalcJob metadata

Step 2 → Send ONLY relevant snippets to Llama 3:
          - Max ~500 tokens of actual log content
          - Plus exit_status for context

Step 3 → ChromaDB cross-reference:
          - Match against indexed known fixes
          - Return actionable suggestions
```

Example output:
```
=== Diagnostic Report for pk=1234 ===

Your calculation (pk=1234) failed with exit code 410.
Cause: SCF convergence not reached within the maximum number of iterations.

Suggested fixes:
  1. Increase electron_maxstep from default (100) to 200 or higher.
  2. Reduce conv_thr from 1e-8 to 1e-6 for less strict convergence.
  3. Try a different mixing_mode (e.g. 'local-TF' instead of 'plain').
  4. Reduce mixing_beta to 0.3 or lower to stabilise SCF mixing.
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up AiiDA profile (if not already done)
verdi quicksetup

# 3. Install and start Ollama
# https://ollama.ai
ollama pull llama3
ollama serve

# 4. Index error patterns into ChromaDB
python agents/search.py

# 5. Start the MCP server
python mcp_server/server.py

# 6. Run the natural language interface
python llm/ollama_agent.py
```

---

## Example Interactions

```
You: Why did my last calculation fail?

[Calling tool: diagnose_calculation({'pk': 1234})]

AiiDA Assistant: Your calculation failed with exit code 410 — the SCF 
cycle did not converge. Try increasing electron_maxstep to 200 and 
reducing conv_thr from 1e-8 to 1e-6.
```

```
You: How do I launch a PwCalculation?

[Calling tool: search_docs({'query': 'how to launch PwCalculation'})]

AiiDA Assistant: To launch a PwCalculation, use the builder pattern...
```

---

## What the Full GSoC Project Adds

This PoC is intentionally minimal. The full project will add:

- **LangGraph orchestration** — stateful multi-step agent workflows
- **Execution Agent** — safe calculation launching with human-in-the-loop confirmation
- **Full AiiDA docs indexing** — scraping and indexing docs.aiida.net into ChromaDB
- **Daemon polling** — real-time monitoring of running calculations
- **Diagnostic benchmark** — 10 real failed CalcJobs with known root causes (target: ≥8/10 correct)
- **Gradio UI** — browser interface for non-technical users

---

## Author

**Aryan Srivastava** — [github.com/aryansri05](https://github.com/aryansri05)

GSoC 2026 applicant — AiiDA Natural Language Interface (NumFOCUS)
