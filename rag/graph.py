"""
graph.py – Lightweight GraphRAG using NetworkX.

Builds an in-memory graph where every retrieved chunk is a node and
edges connect chunks that share significant keyword overlap (measured
via cosine similarity on TF-IDF-style keyword vectors).  Given a query
the graph is used to surface *related* chunks that would otherwise be
missed by a flat top-k retrieval.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Set, Tuple

import networkx as nx
from langchain_core.documents import Document

from rag.retriever import DocumentRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SIMILARITY_THRESHOLD: float = 0.15
_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "it", "its", "this",
    "that", "these", "those", "i", "you", "he", "she", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "our", "their", "mine",
    "yours", "hers", "ours", "theirs", "what", "which", "who", "whom",
    "whose", "where", "when", "why", "how", "not", "no", "nor", "and",
    "but", "or", "so", "if", "then", "than", "too", "very", "just", "about",
    "above", "after", "again", "all", "also", "am", "any", "as", "at",
    "because", "before", "below", "between", "both", "by", "d", "don",
    "down", "during", "each", "few", "for", "from", "further", "get",
    "got", "here", "in", "into", "more", "most", "of", "off", "on",
    "once", "only", "other", "out", "over", "own", "re", "s", "same",
    "some", "such", "t", "to", "under", "until", "up", "ve", "with",
}

_WORD_PATTERN = re.compile(r"[a-zA-Z]{3,}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_keywords(text: str) -> Counter:
    """Extract keyword frequencies from *text*, filtering stopwords."""
    words = _WORD_PATTERN.findall(text.lower())
    return Counter(w for w in words if w not in _STOPWORDS)


def _cosine_similarity(vec_a: Counter, vec_b: Counter) -> float:
    """Compute cosine similarity between two :pyclass:`Counter` vectors."""
    if not vec_a or not vec_b:
        return 0.0

    common_keys = set(vec_a.keys()) & set(vec_b.keys())
    if not common_keys:
        return 0.0

    dot = sum(vec_a[k] * vec_b[k] for k in common_keys)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# GraphRAG
# ---------------------------------------------------------------------------

class GraphRAG:
    """Graph-based retrieval augmentation using NetworkX.

    Parameters
    ----------
    retriever:
        A :pyclass:`DocumentRetriever` instance.  If ``None`` a default
        retriever with default settings is created.
    similarity_threshold:
        Minimum cosine-similarity score to create an edge between two
        chunk nodes.
    """

    def __init__(
        self,
        retriever: DocumentRetriever | None = None,
        similarity_threshold: float = _SIMILARITY_THRESHOLD,
    ) -> None:
        self._retriever: DocumentRetriever = retriever or DocumentRetriever()
        self._threshold: float = similarity_threshold

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(self, documents: List[Document]) -> nx.Graph:
        """Build a keyword-similarity graph from *documents*.

        Each document becomes a node.  An edge is added between two nodes
        when their keyword-vector cosine similarity exceeds the configured
        threshold.

        Parameters
        ----------
        documents:
            LangChain ``Document`` objects (chunks).

        Returns
        -------
        nx.Graph
            The constructed graph.
        """
        graph = nx.Graph()

        # Pre-compute keyword vectors for every chunk.
        keyword_vectors: List[Counter] = []
        for idx, doc in enumerate(documents):
            kw = _extract_keywords(doc.page_content)
            keyword_vectors.append(kw)
            graph.add_node(
                idx,
                content=doc.page_content,
                metadata=doc.metadata,
                keywords=kw,
            )

        # Create edges for chunks that are sufficiently similar.
        n = len(documents)
        for i in range(n):
            for j in range(i + 1, n):
                sim = _cosine_similarity(keyword_vectors[i], keyword_vectors[j])
                if sim >= self._threshold:
                    graph.add_edge(i, j, weight=sim)

        logger.info(
            "Graph built: %d nodes, %d edges (threshold=%.2f)",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            self._threshold,
        )
        return graph

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _rank_nodes(self, graph: nx.Graph, seed_indices: List[int]) -> List[int]:
        """Rank nodes by their connectivity to the *seed* nodes.

        Nodes are scored by the sum of edge weights connecting them to
        any seed node plus a bonus from PageRank centrality.  Already-
        seen seed nodes are kept but scored lower than newly discovered
        neighbours.

        Returns a list of node indices ordered by descending score.
        """
        if graph.number_of_nodes() == 0:
            return []

        # PageRank gives a global importance signal.
        try:
            pagerank: Dict[int, float] = nx.pagerank(graph, weight="weight")
        except nx.NetworkXError:
            pagerank = {n: 1.0 / graph.number_of_nodes() for n in graph.nodes}

        scores: Dict[int, float] = {}
        seed_set = set(seed_indices)

        for seed in seed_indices:
            if seed not in graph:
                continue
            for neighbour in graph.neighbors(seed):
                edge_w: float = graph[seed][neighbour].get("weight", 0.0)
                scores[neighbour] = scores.get(neighbour, 0.0) + edge_w

        # Merge pagerank into scores for all nodes with connections.
        for node in scores:
            scores[node] += pagerank.get(node, 0.0)

        # Include seed nodes themselves (lower bonus so new context ranks higher).
        for seed in seed_indices:
            if seed in graph:
                scores.setdefault(seed, 0.0)
                scores[seed] += pagerank.get(seed, 0.0) * 0.5

        ranked = sorted(scores, key=lambda n: scores[n], reverse=True)
        return ranked

    def query_graph(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve chunks via the graph and return an enriched context.

        Steps
        -----
        1. Retrieve top-*k* chunks from the vector store.
        2. Build a graph from those chunks **plus** additional related
           chunks (fetched at ``2 × k``).
        3. Rank nodes by graph connectivity and PageRank.
        4. Return the highest-ranked chunks as ``Document`` objects.

        Parameters
        ----------
        query:
            Natural-language question.
        k:
            Number of enriched chunks to return.

        Returns
        -------
        List[Document]
        """
        # Retrieve an expanded pool of candidates.
        expanded_k = min(k * 3, 20)
        pool: List[Document] = self._retriever.retrieve(query, k=expanded_k)

        if not pool:
            logger.warning("No chunks retrieved for graph query: %s", query)
            return []

        # Build graph over the candidate pool.
        graph = self.build_graph(pool)

        # Identify the seed nodes (the original top-k).
        seed_indices: List[int] = list(range(min(k, len(pool))))

        # Rank all nodes relative to the seeds.
        ranked_indices = self._rank_nodes(graph, seed_indices)

        # Collect the top-k enriched documents.
        enriched: List[Document] = []
        seen: set[int] = set()
        for idx in ranked_indices:
            if idx in seen:
                continue
            seen.add(idx)
            node_data = graph.nodes[idx]
            enriched.append(
                Document(
                    page_content=node_data["content"],
                    metadata=node_data["metadata"],
                )
            )
            if len(enriched) >= k:
                break

        # If graph ranking returned fewer than k, pad with remaining seeds.
        for idx in seed_indices:
            if len(enriched) >= k:
                break
            if idx not in seen:
                seen.add(idx)
                enriched.append(pool[idx])

        logger.info(
            "GraphRAG returned %d enriched chunks for: %.80s…",
            len(enriched),
            query,
        )
        return enriched

    def get_related_chunks(self, query: str, k: int = 4) -> List[Document]:
        """Convenience alias for :pymeth:`query_graph`.

        Parameters
        ----------
        query:
            Natural-language question.
        k:
            Number of results to return.

        Returns
        -------
        List[Document]
        """
        return self.query_graph(query, k=k)

    # ------------------------------------------------------------------
    # Inspection utilities
    # ------------------------------------------------------------------

    def get_graph_stats(self, documents: List[Document]) -> Dict[str, object]:
        """Return basic statistics about the graph built from *documents*.

        Useful for debugging and monitoring.
        """
        graph = self.build_graph(documents)
        components = list(nx.connected_components(graph))
        return {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "connected_components": len(components),
            "largest_component_size": max(len(c) for c in components) if components else 0,
            "density": nx.density(graph),
        }
