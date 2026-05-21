"""Memory manager - unified facade over working and long-term memory."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from agentic_os.core.graph.edge import EdgeType
from agentic_os.core.graph.node import MemoryNode
from agentic_os.core.memory.consolidation import simple_consolidation
from agentic_os.core.memory.longterm import LongTermMemory, RetrievalStrategy
from agentic_os.core.memory.working import WorkingMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """Unified memory facade combining short-term and long-term memory.

    Provides high-level operations for adding experiences, recalling,
    reflecting, consolidating, and forgetting. Agents should interact
    with memory exclusively through this class.
    """

    def __init__(self, working_capacity: int = 50) -> None:
        """Initialize the memory manager.

        Args:
            working_capacity: Maximum items in working (short-term) memory.
        """
        self.working = WorkingMemory(capacity=working_capacity)
        self.longterm = LongTermMemory()

    def __repr__(self) -> str:
        return f"MemoryManager(working={self.working.size}, longterm={self.longterm.graph.node_count})"

    def add_experience(self, content: str, **metadata: Any) -> str:
        """Add an experience to working memory.

        Args:
            content: Textual description of the experience.
            **metadata: Arbitrary metadata to attach.

        Returns:
            The auto-generated working-memory key.
        """
        key = self.working.put_content(content, **metadata)
        logger.debug("add_experience: stored as %s", key)
        return key

    def store_fact(self, content: str, importance: float = 0.6, **metadata: Any) -> str:
        """Store a fact directly in long-term memory.

        Args:
            content: Fact text.
            importance: Initial importance score in [0, 1].
            **metadata: Additional metadata.

        Returns:
            The new long-term memory node ID.
        """
        node_id = self.longterm.store_fact(content, importance, **metadata)
        logger.debug("store_fact: %s", node_id)
        return node_id

    def store_reflection(self, content: str, importance: float = 0.7, **metadata: Any) -> str:
        """Store a reflection directly in long-term memory.

        Args:
            content: Reflection text.
            importance: Initial importance score in [0, 1].
            **metadata: Additional metadata.

        Returns:
            The new long-term memory node ID.
        """
        node_id = self.longterm.store_reflection(content, importance, **metadata)
        logger.debug("store_reflection: %s", node_id)
        return node_id

    def recall(self, query: str, top_k: int = 10) -> list[MemoryNode]:
        """Recall memories matching *query* from both layers.

        The method employs a multi-strategy retrieval pipeline:

        1. **Working-memory scan** - performs a case-insensitive substring
           match on all working-memory entries.
        2. **Long-term keyword search** - queries the knowledge-graph
           inverted index via ``RetrievalStrategy.KEYWORD``.

        Results from both layers are merged and deduplicated before
        being returned.

        Args:
            query: Search text used for matching.
            top_k: Maximum number of results.

        Returns:
            Deduplicated list of ``MemoryNode`` instances, up to *top_k*.

        Examples:
            >>> mm = MemoryManager()
            >>> mm.add_experience("deployed v2.0 to production")
            >>> mm.store_fact("production runs on k8s")
            >>> mm.recall("production", top_k=5)
        """
        results: list[MemoryNode] = []
        from agentic_os.core.graph.node import create_episode
        q_lower = query.lower()

        # Search from working memory
        for _key, entry in self.working.peek_all():
            content = entry.get("content", "")
            if q_lower in content.lower():
                results.append(create_episode(content, **entry.get("metadata", {})))

        # Retrieve from long-term memory
        lt_results = self.longterm.retrieve(query, RetrievalStrategy.KEYWORD, top_k)
        seen_ids = {r.id for r in results}
        for node in lt_results:
            if node.id not in seen_ids:
                results.append(node)
                seen_ids.add(node.id)

        return results[:top_k]

    def recall_by_importance(self, top_k: int = 10) -> list[MemoryNode]:
        """Recall the most important memories from long-term storage.

        Args:
            top_k: Maximum number of results.

        Returns:
            List of ``MemoryNode`` sorted by descending importance.
        """
        return self.longterm.retrieve("", RetrievalStrategy.IMPORTANCE, top_k)

    def recall_by_recency(self, top_k: int = 10) -> list[MemoryNode]:
        """Recall the most recently stored memories.

        Args:
            top_k: Maximum number of results.

        Returns:
            List of ``MemoryNode`` ordered from newest to oldest.
        """
        return self.longterm.retrieve("", RetrievalStrategy.RECENCY, top_k)

    def recall_associated(self, seed_id: str, top_k: int = 10) -> list[MemoryNode]:
        """Recall memories associated with a seed node via graph traversal.

        Performs a BFS from *seed_id* up to depth 2.

        Args:
            seed_id: The starting node ID.
            top_k: Maximum number of results.

        Returns:
            List of ``MemoryNode`` reachable from the seed.
        """
        return self.longterm.retrieve("", RetrievalStrategy.ASSOCIATION, top_k, seed_id=seed_id)

    def link_memories(self, source_id: str, target_id: str,
                      edge_type: EdgeType = EdgeType.ASSOCIATIVE,
                      weight: float = 1.0) -> None:
        """Create an association between two long-term memories.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            edge_type: Semantic type of the edge.
            weight: Edge weight.
        """
        self.longterm.link(source_id, target_id, edge_type, weight)

    def consolidate(self) -> int:
        """Move all working-memory entries into long-term memory.

        After consolidation the working memory is cleared.

        Returns:
            Number of entries migrated.
        """
        count = simple_consolidation(self.working, self.longterm)
        logger.info("consolidate: migrated %d entries to long-term", count)
        return count

    def reflect(self, pattern_fn: Callable[[list[tuple[str, dict[str, Any]]]], list[str]] | None = None) -> list[str]:
        """Trigger reflection: extract patterns from working memory and generate long-term reflection nodes.

        Args:
            pattern_fn: Optional custom pattern extraction function
                        (working_items) -> list[str]
                        Defaults to simple frequency statistics.
        """
        items = self.working.peek_all()
        if not items:
            return []

        insights = pattern_fn(items) if pattern_fn else self._default_reflection(items)

        reflection_ids = []
        for insight in insights:
            rid = self.longterm.store_reflection(insight, importance=0.7)
            reflection_ids.append(rid)
        logger.info("reflect: generated %d insights from %d working items", len(reflection_ids), len(items))
        return reflection_ids

    def forget(self, min_importance: float = 0.1) -> int:
        """Remove long-term memories whose importance is below *min_importance*.

        Args:
            min_importance: Threshold; nodes with importance < this are removed.

        Returns:
            Number of nodes removed.
        """
        graph = self.longterm.graph
        to_remove = [
            node.id for node in graph.nodes
            if node.importance < min_importance
        ]
        count = 0
        for nid in to_remove:
            if graph.remove_node(nid) is not None:
                count += 1
        return count

    def stats(self) -> dict[str, Any]:
        """Return summary statistics for both memory layers.

        Returns:
            Dict with ``working_memory`` (int) and ``longterm_memory`` (dict).
        """
        return {
            "working_memory": self.working.size,
            "longterm_memory": self.longterm.stats(),
        }

    def _default_reflection(self, items: list[tuple[str, dict[str, Any]]]) -> list[str]:
        """Extract high-frequency keywords from working memory as reflection.

        Args:
            items: List of ``(key, entry)`` tuples from working memory.

        Returns:
            List of summary strings (one per reflection).
        """
        from collections import Counter

        word_count: Counter[str] = Counter()
        for _, entry in items:
            content = entry.get("content", "")
            words = content.lower().split()
            word_count.update(w for w in words if len(w) > 2)

        top_words = word_count.most_common(5)
        if not top_words:
            return []

        summary_words = ", ".join(w for w, _ in top_words)
        return [f"Recent topics of interest: {summary_words} ({len(items)} experiences)"]
