"""
Search Agent — ChromaDB RAG
Indexes AiiDA documentation and error patterns for semantic search.
Used by the Diagnostic Agent (cross-reference fixes) and
directly by users asking "how do I...?" questions.
"""

import chromadb
from chromadb.utils import embedding_functions

# Use a lightweight local embedding model — no API calls needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_DOCS = "aiida_docs"
COLLECTION_ERRORS = "aiida_error_patterns"


def get_client() -> chromadb.Client:
    """Return a persistent ChromaDB client stored locally."""
    return chromadb.PersistentClient(path="./.chromadb")


def get_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def index_error_patterns(client: chromadb.Client) -> None:
    """
    Pre-index known AiiDA CalcJob error patterns and their fixes.
    This is Step 3 of the Diagnostic Agent — cross-reference against
    indexed known fixes rather than relying on LLM parametric knowledge.
    """
    ef = get_embedding_fn()
    collection = client.get_or_create_collection(
        name=COLLECTION_ERRORS,
        embedding_function=ef,
    )

    # Curated error pattern knowledge base
    # In production this will be expanded from community-reported issues
    patterns = [
        {
            "id": "exit_410_scf",
            "document": "SCF convergence not achieved exit code 410 electron_maxstep conv_thr mixing",
            "metadata": {
                "exit_code": 410,
                "fix": "Increase electron_maxstep to 200. Reduce conv_thr from 1e-8 to 1e-6. Try mixing_mode local-TF.",
            },
        },
        {
            "id": "exit_310_walltime",
            "document": "calculation stopped walltime limit reached exit code 310 scheduler timeout",
            "metadata": {
                "exit_code": 310,
                "fix": "Request more walltime. Reduce system size or use k-point parallelisation.",
            },
        },
        {
            "id": "exit_320_memory",
            "document": "out of memory OOM exit code 320 memory allocation failed",
            "metadata": {
                "exit_code": 320,
                "fix": "Increase memory per node. Reduce ecutwfc to lower memory footprint.",
            },
        },
        {
            "id": "exit_420_ionic",
            "document": "ionic convergence not reached exit code 420 relaxation nstep forc_conv_thr",
            "metadata": {
                "exit_code": 420,
                "fix": "Increase nstep. Adjust forc_conv_thr. Switch to a more stable ion_dynamics algorithm.",
            },
        },
        {
            "id": "exit_411_bfgs",
            "document": "BFGS history limit exceeded exit code 411 geometry optimisation failed",
            "metadata": {
                "exit_code": 411,
                "fix": "Increase bfgs_ndim. Switch to ion_dynamics = damp for more stable relaxation.",
            },
        },
    ]

    existing = set(collection.get()["ids"])
    new_patterns = [p for p in patterns if p["id"] not in existing]

    if new_patterns:
        collection.add(
            ids=[p["id"] for p in new_patterns],
            documents=[p["document"] for p in new_patterns],
            metadatas=[p["metadata"] for p in new_patterns],
        )
        print(f"Indexed {len(new_patterns)} new error patterns.")
    else:
        print("Error pattern index already up to date.")


def search_error_patterns(query: str, n_results: int = 3) -> list[dict]:
    """
    Semantic search over indexed error patterns.
    Called by the Diagnostic Agent after local keyword extraction.
    Returns the closest matching known fixes.
    """
    client = get_client()
    ef = get_embedding_fn()
    collection = client.get_or_create_collection(
        name=COLLECTION_ERRORS,
        embedding_function=ef,
    )

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
    )

    matches = []
    for i, doc in enumerate(results["documents"][0]):
        matches.append({
            "document": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return matches


def index_docs_chunk(chunks: list[dict]) -> None:
    """
    Index AiiDA documentation chunks into ChromaDB.
    Each chunk should have: id, text, metadata (source, section).
    In production, this is populated by scraping docs.aiida.net.
    """
    client = get_client()
    ef = get_embedding_fn()
    collection = client.get_or_create_collection(
        name=COLLECTION_DOCS,
        embedding_function=ef,
    )

    existing = set(collection.get()["ids"])
    new_chunks = [c for c in chunks if c["id"] not in existing]

    if new_chunks:
        collection.add(
            ids=[c["id"] for c in new_chunks],
            documents=[c["text"] for c in new_chunks],
            metadatas=[c.get("metadata", {}) for c in new_chunks],
        )


def search_docs(query: str, n_results: int = 3) -> list[dict]:
    """
    Semantic search over AiiDA documentation.
    Used by the Search Agent to answer how-to questions.
    """
    client = get_client()
    ef = get_embedding_fn()
    collection = client.get_or_create_collection(
        name=COLLECTION_DOCS,
        embedding_function=ef,
    )

    if collection.count() == 0:
        return [{"document": "Documentation index is empty. Run indexing first.", "metadata": {}}]

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
    )

    return [
        {"document": doc, "metadata": results["metadatas"][0][i]}
        for i, doc in enumerate(results["documents"][0])
    ]


if __name__ == "__main__":
    client = get_client()
    index_error_patterns(client)

    print("\nTest search: 'SCF did not converge'")
    results = search_error_patterns("SCF did not converge after 100 iterations")
    for r in results:
        print(f"  Match (distance={r['distance']:.3f}): {r['metadata'].get('fix')}")
