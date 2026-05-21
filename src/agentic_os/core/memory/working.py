"""Working memory (short-term) backed by an LRU cache built on OrderedDict."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

from agentic_os.utils.hashing import content_id

logger = logging.getLogger(__name__)


class WorkingMemory:
    """Short-term working memory with LRU eviction policy.

    Recently accessed items are retained; the least-recently-used item is
    evicted when capacity is exceeded. All operations are O(1).

    Use this for transient agent context that does not need to persist
    across sessions.
    """

    def __init__(self, capacity: int = 50) -> None:
        """Initialize the working memory store.

        Args:
            capacity: Maximum number of items to retain. When exceeded,
                the least-recently-used item is evicted.
        """
        self.capacity = capacity
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def put(self, key: str, value: dict[str, Any]) -> str:
        """Store an item and refresh its LRU position.

        If the key already exists its value is updated and moved to the end.
        If capacity is exceeded the oldest item is evicted first.

        Args:
            key: Unique identifier for the memory entry.
            value: Payload dict to store.

        Returns:
            The same *key* that was passed in.

        Examples:
            >>> wm = WorkingMemory(capacity=3)
            >>> wm.put("a", {"content": "hello"})
            'a'
        """
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if len(self._store) >= self.capacity:
                evicted_key, _ = self._store.popitem(last=False)
                logger.warning("put: evicted key %r (capacity=%d)", evicted_key, self.capacity)
        self._store[key] = value
        logger.debug("put: stored key %r", key)
        return key

    def put_content(self, content: str, **metadata: Any) -> str:
        """Store content with an auto-generated key derived from its hash.

        Args:
            content: Text content to store.
            **metadata: Arbitrary metadata attached to the entry.

        Returns:
            The auto-generated key (prefixed with ``wm_``).

        Examples:
            >>> wm = WorkingMemory()
            >>> wm.put_content("hello world", source="user")
            'wm_...'
        """
        key = content_id(content, prefix="wm")
        entry = {"content": content, "metadata": metadata}
        return self.put(key, entry)

    def get(self, key: str) -> dict[str, Any] | None:
        """Retrieve an item and refresh its LRU position.

        Args:
            key: Identifier of the entry to retrieve.

        Returns:
            The stored value dict, or ``None`` if the key does not exist.
        """
        value = self._store.get(key)
        if value is not None:
            self._store.move_to_end(key)
        logger.debug("get: key %r found=%s", key, value is not None)
        return value

    def remove(self, key: str) -> dict[str, Any] | None:
        """Remove and return an entry without affecting LRU order of others.

        Args:
            key: Identifier of the entry to remove.

        Returns:
            The removed value dict, or ``None`` if not found.
        """
        return self._store.pop(key, None)

    def peek_all(self) -> list[tuple[str, dict[str, Any]]]:
        """Return all entries without affecting LRU order.

        Returns:
            List of ``(key, value)`` tuples in insertion order.
        """
        return list(self._store.items())

    @property
    def size(self) -> int:
        """int: Current number of entries in the store."""
        return len(self._store)

    def clear(self) -> None:
        """Remove all entries from the store."""
        self._store.clear()

    def recent(self, n: int = 10) -> list[tuple[str, dict[str, Any]]]:
        """Return the *n* most recently used entries.

        Args:
            n: Maximum number of entries to return.

        Returns:
            List of ``(key, value)`` tuples ordered from oldest to newest.
        """
        items = list(self._store.items())
        return items[-n:]

    def __contains__(self, key: str) -> bool:
        """Check whether a key exists in the store."""
        return key in self._store

    def __len__(self) -> int:
        """Return the number of stored entries."""
        return self.size
