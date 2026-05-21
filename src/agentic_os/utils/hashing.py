"""Content hashing utilities based on the FNV-1a 64-bit algorithm."""

from __future__ import annotations

# FNV-1a 64-bit parameters
_FNV_OFFSET_BASIS = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK = (1 << 64) - 1


def fnv1a_hash(data: str | bytes) -> int:
    """Compute a deterministic FNV-1a 64-bit hash.

    Args:
        data: Input string or bytes. Strings are encoded as UTF-8.

    Returns:
        Unsigned 64-bit integer hash value.

    Examples:
        >>> fnv1a_hash("hello")
        0xa430d84680aabd0b
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    h = _FNV_OFFSET_BASIS
    for byte in data:
        h ^= byte
        h = (h * _FNV_PRIME) & _MASK
    return h


def content_id(content: str, prefix: str = "") -> str:
    """Generate a content-derived unique ID as a 16-char hex string.

    Args:
        content: Source text to hash.
        prefix: Optional prefix prepended to the hex digest (e.g. ``"wm"``).

    Returns:
        Hex string of the form ``{prefix}_{hex16}`` or just ``hex16``.

    Examples:
        >>> content_id("hello", prefix="goal")
        'goal_a430d84680aabd0b'
    """
    h = fnv1a_hash(content)
    hex_str = f"{h:016x}"
    return f"{prefix}_{hex_str}" if prefix else hex_str


def combined_id(*parts: str) -> str:
    """Generate a unique ID by hashing concatenated parts.

    Args:
        *parts: Strings to join (with ``|`` separator) and hash.

    Returns:
        16-char hex string derived from the combined content.

    Examples:
        >>> combined_id("user", "session", "42")
    """
    return content_id("|".join(parts))
