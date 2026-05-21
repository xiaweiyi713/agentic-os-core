"""Memory consolidation strategies - mimicking human sleep-time memory consolidation."""

from __future__ import annotations

import logging
from typing import Any

from agentic_os.core.memory.longterm import LongTermMemory
from agentic_os.core.memory.working import WorkingMemory

logger = logging.getLogger(__name__)


def simple_consolidation(working: WorkingMemory, longterm: LongTermMemory) -> int:
    """Migrate all working-memory entries into long-term memory as episodes.

    After migration the working memory is cleared.

    Args:
        working: The source working-memory instance.
        longterm: The destination long-term-memory instance.

    Returns:
        Number of entries migrated.

    Examples:
        >>> simple_consolidation(wm, ltm)
        42
    """
    items = working.peek_all()
    count = 0
    for _key, entry in items:
        content = entry.get("content", "")
        metadata = entry.get("metadata", {})
        if content:
            longterm.store_episode(content, **metadata)
            count += 1
    working.clear()
    logger.info("simple_consolidation: migrated %d entries", count)
    return count


def importance_consolidation(working: WorkingMemory, longterm: LongTermMemory,
                             threshold: float = 0.3) -> int:
    """Migrate only working-memory entries whose importance >= *threshold*.

    All working-memory entries are cleared regardless of whether they were
    migrated.

    Args:
        working: The source working-memory instance.
        longterm: The destination long-term-memory instance.
        threshold: Minimum importance (from metadata) to keep. Defaults to 0.3.

    Returns:
        Number of entries migrated.
    """
    items = working.peek_all()
    count = 0
    for _key, entry in items:
        content = entry.get("content", "")
        importance = entry.get("metadata", {}).get("importance", 0.5)
        if content and importance >= threshold:
            longterm.store_episode(content, **entry.get("metadata", {}))
            count += 1
    working.clear()
    logger.info("importance_consolidation: migrated %d entries", count)
    return count


def pattern_consolidation(working: WorkingMemory, longterm: LongTermMemory) -> int:
    """Group similar memories and generate abstract reflection nodes.

    Memories sharing more than 30 % of their word tokens are grouped.
    Groups of 2+ items produce a reflection node summarising shared keywords;
    singletons are migrated directly as episodes.

    Args:
        working: The source working-memory instance.
        longterm: The destination long-term-memory instance.

    Returns:
        Number of individual entries migrated (reflections excluded).
    """
    items = working.peek_all()
    if not items:
        return 0

    # Group by content similarity (simple keyword overlap)
    groups: list[list[tuple[str, dict[str, Any]]]] = []
    assigned: set[str] = set()

    for key, entry in items:
        if key in assigned:
            continue
        group = [(key, entry)]
        assigned.add(key)
        content_words = set(entry.get("content", "").lower().split())

        for other_key, other_entry in items:
            if other_key in assigned:
                continue
            other_words = set(other_entry.get("content", "").lower().split())
            if content_words and other_words:
                overlap = len(content_words & other_words) / max(len(content_words), 1)
                if overlap > 0.3:
                    group.append((other_key, other_entry))
                    assigned.add(other_key)

        groups.append(group)

    # Generate abstract reflection nodes for each group
    count = 0
    for group in groups:
        if len(group) < 2:
            # Single memory: migrate directly
            _, entry = group[0]
            content = entry.get("content", "")
            if content:
                longterm.store_episode(content, **entry.get("metadata", {}))
                count += 1
        else:
            # Multiple similar memories: extract common keywords as reflection
            all_words: set[str] = set()
            for _, entry in group:
                all_words.update(entry.get("content", "").lower().split())
            # Take high-frequency words
            common = [w for w in all_words if len(w) > 2][:5]
            if common:
                summary = f"Pattern detected: {len(group)} experiences involving {', '.join(common)}"
                longterm.store_reflection(summary, importance=0.7,
                                          group_size=len(group))
            # Also keep original experiences
            for _, entry in group:
                content = entry.get("content", "")
                if content:
                    longterm.store_episode(content, **entry.get("metadata", {}))
                    count += 1

    working.clear()
    logger.info("pattern_consolidation: migrated %d entries", count)
    return count
