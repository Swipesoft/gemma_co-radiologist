"""
athena_dda.retriever — BM25 Tool Retriever
════════════════════════════════════════════
Optional BM25-based tool recommender for dynamic tool selection.
"""

from typing import Dict, List, Optional


class BM25ToolRetriever:
    """Retrieves relevant tools from a registry using BM25 text similarity."""

    def __init__(self):
        self._bm25 = None
        self._tool_names: List[str] = []
        self._tools: Dict = {}

    def rebuild_index(self, tools: Dict):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print("[BM25ToolRetriever] rank_bm25 not installed. Retriever disabled.")
            return

        self._tools = tools
        self._tool_names = list(tools.keys())
        docs = []
        for name in self._tool_names:
            tool = tools[name]
            # Build a document from tool metadata
            doc = getattr(tool, "get_bm25_document", None)
            if doc and callable(doc):
                docs.append(doc())
            else:
                # Fallback: use name + description
                parts = [name, getattr(tool, "description", "")]
                parts += list(getattr(tool, "arg_sources", {}).values())
                parts += list(getattr(tool, "output_keys", {}).keys())
                docs.append(" ".join(parts))
        self._bm25 = BM25Okapi([d.lower().split() for d in docs])

    def retrieve(self, query: str, top_k: int = 5,
                 exclude: Optional[List[str]] = None, min_score: float = 0.3):
        exclude = exclude or []
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        valid = [
            (i, scores[i])
            for i in range(len(self._tool_names))
            if scores[i] >= min_score and self._tool_names[i] not in exclude
        ]
        valid.sort(key=lambda x: x[1], reverse=True)
        return [self._tools[self._tool_names[i]] for i, _ in valid[:top_k]]
