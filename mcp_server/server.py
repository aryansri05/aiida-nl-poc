"""
AiiDA-MCP Server
Exposes AiiDA's API as structured tools for local LLM interaction.
No LLM ever touches AiiDA directly — it only calls these safe, typed tools.
"""

from fastmcp import FastMCP
from aiida import load_profile
from aiida.orm import load_node, QueryBuilder, CalcJobNode, Node
from aiida.engine import ProcessState

load_profile()

mcp = FastMCP("aiida-mcp")


@mcp.tool()
def query_node(pk: int) -> dict:
    """
    Query an AiiDA node by its PK and return its key attributes.
    Use this when the user asks about a specific node or calculation.
    """
    node = load_node(pk)
    return {
        "pk": node.pk,
        "uuid": str(node.uuid),
        "node_type": node.node_type,
        "label": node.label,
        "description": node.description,
        "attributes": node.attributes,
        "ctime": str(node.ctime),
        "mtime": str(node.mtime),
    }


@mcp.tool()
def get_failed_calculations(limit: int = 10) -> list[dict]:
    """
    Return recently failed CalcJobs with their exit status and labels.
    Use this when the user asks why calculations failed or wants to see failures.
    """
    qb = QueryBuilder()
    qb.append(
        CalcJobNode,
        filters={"attributes.exit_status": {">": 0}},
        project=["pk", "uuid", "label", "attributes.exit_status", "ctime"],
    )
    qb.order_by({CalcJobNode: {"ctime": "desc"}})
    qb.limit(limit)

    results = []
    for pk, uuid, label, exit_status, ctime in qb.all():
        results.append({
            "pk": pk,
            "uuid": str(uuid),
            "label": label or "(no label)",
            "exit_status": exit_status,
            "ctime": str(ctime),
        })
    return results


@mcp.tool()
def poll_calculation_status(pk: int) -> dict:
    """
    Return the current process state of a CalcJob.
    Use this to check if a running calculation has finished, failed, or is still active.
    Note: Full daemon polling integration completed in Week 10 of GSoC.
    """
    node = load_node(pk)
    if not isinstance(node, CalcJobNode):
        return {"error": f"Node {pk} is not a CalcJobNode (got {node.node_type})"}

    return {
        "pk": pk,
        "process_state": node.process_state.value if node.process_state else "unknown",
        "exit_status": node.exit_status,
        "exit_message": node.exit_message,
        "is_finished": node.is_finished,
        "is_failed": node.is_failed,
    }


if __name__ == "__main__":
    mcp.run()
