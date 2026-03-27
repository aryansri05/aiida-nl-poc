"""
Ollama Agent — Llama 3 Tool Calling
Ties everything together: natural language in → MCP tool calls out.
Runs fully locally via Ollama. No API keys, no internet required.
This is why the system works on HPC clusters with restricted network access.
"""

import json
import re
import requests
from agents.diagnostic import diagnose_calculation, format_diagnostic_output
from agents.search import search_docs, search_error_patterns

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3"

# Tool definitions — these map to our MCP tools and agent functions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "diagnose_calculation",
            "description": "Diagnose why an AiiDA CalcJob failed. Returns exit code, cause, and suggested fixes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pk": {"type": "integer", "description": "The PK of the failed CalcJob node"}
                },
                "required": ["pk"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search AiiDA documentation to answer how-to questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question to search for"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_error_patterns",
            "description": "Search known AiiDA error patterns for fix suggestions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Error description or symptoms"}
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are an AiiDA assistant. AiiDA is a scientific workflow management framework used in computational materials science.

You help researchers interact with their AiiDA database using natural language.

You have access to the following tools:
- diagnose_calculation(pk): Diagnose a failed CalcJob by its PK
- search_docs(query): Search AiiDA documentation
- search_error_patterns(query): Find known fixes for AiiDA errors

Always use tools to answer questions about AiiDA data. Never guess PKs or make up results.
Be concise and actionable in your responses."""


def call_tool(name: str, args: dict) -> str:
    """Execute a tool call and return the result as a string."""
    if name == "diagnose_calculation":
        result = diagnose_calculation(args["pk"])
        return format_diagnostic_output(result)

    elif name == "search_docs":
        results = search_docs(args["query"])
        if not results:
            return "No relevant documentation found."
        return "\n\n".join(
            f"[{r['metadata'].get('source', 'docs')}]\n{r['document']}"
            for r in results
        )

    elif name == "search_error_patterns":
        results = search_error_patterns(args["query"])
        if not results:
            return "No matching error patterns found."
        return "\n\n".join(
            f"Fix suggestion: {r['metadata'].get('fix', 'No fix available')}"
            for r in results
        )

    return f"Unknown tool: {name}"


def extract_tool_call(response_text: str) -> tuple[str, dict] | None:
    """
    Parse tool call from Llama 3 response.
    Llama 3 tool calls come back as JSON in the message content.
    """
    try:
        # Look for JSON tool call pattern in response
        match = re.search(r'\{"name":\s*"(\w+)",\s*"parameters":\s*(\{.*?\})\}', response_text, re.DOTALL)
        if match:
            name = match.group(1)
            params = json.loads(match.group(2))
            return name, params
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


def chat(user_message: str) -> str:
    """
    Main chat loop. Sends user message to Llama 3 via Ollama,
    handles tool calls, and returns the final response.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # First call — let Llama 3 decide if a tool is needed
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "messages": messages, "tools": TOOLS, "stream": False},
    )
    response.raise_for_status()
    data = response.json()

    message = data["message"]
    tool_calls = message.get("tool_calls", [])

    # Handle tool calls if Llama 3 decided to use one
    if tool_calls:
        messages.append(message)

        for tool_call in tool_calls:
            fn = tool_call["function"]
            tool_name = fn["name"]
            tool_args = fn.get("arguments", {})

            print(f"[Calling tool: {tool_name}({tool_args})]")
            tool_result = call_tool(tool_name, tool_args)

            messages.append({
                "role": "tool",
                "content": tool_result,
            })

        # Second call — Llama 3 synthesizes tool result into final answer
        final_response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "messages": messages, "stream": False},
        )
        final_response.raise_for_status()
        return final_response.json()["message"]["content"]

    # No tool call needed — direct answer
    return message["content"]


if __name__ == "__main__":
    print("AiiDA Natural Language Interface (PoC)")
    print("Model: Llama 3 via Ollama | Tools: MCP + ChromaDB")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        try:
            response = chat(user_input)
            print(f"\nAiiDA Assistant: {response}\n")
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to Ollama. Make sure it's running: `ollama serve`")
        except Exception as e:
            print(f"Error: {e}")
