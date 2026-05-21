"""Serialization utilities - JSON encoding/decoding with dataclass and Enum support."""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from agentic_os.exceptions import ValidationError


class _Encoder(json.JSONEncoder):
    """Custom JSON encoder that handles Enum, datetime, and dataclass objects."""
    def default(self, o: Any) -> Any:
        if isinstance(o, Enum):
            return {"__enum__": f"{type(o).__qualname__}.{o.name}"}
        if isinstance(o, datetime):
            return {"__datetime__": o.isoformat()}
        if is_dataclass(o) and not isinstance(o, type):
            return {"__dataclass__": type(o).__qualname__, "fields": asdict(o)}
        return super().default(o)


def to_json(obj: Any, indent: int = 2) -> str:
    """Serialize an object to a JSON string using the custom encoder.

    Args:
        obj: Any JSON-serialisable object, including dataclasses, Enums,
            and datetime instances.
        indent: Pretty-print indentation level. Defaults to 2.

    Returns:
        JSON string (UTF-8, non-ASCII characters preserved).
    """
    return json.dumps(obj, cls=_Encoder, indent=indent, ensure_ascii=False)


def save_json(obj: Any, path: str | Path) -> None:
    """Serialize and write an object to a JSON file.

    Parent directories are created automatically.

    Args:
        obj: Object to serialize.
        path: Destination file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(to_json(obj), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    """Read and deserialize a JSON file.

    Args:
        path: Source file path.

    Returns:
        Deserialized Python object.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass instance to a JSON-safe dict.

    Handles Enum (converted to its value) and datetime (ISO-format string)
    fields automatically. Nested dataclasses are converted recursively.

    Args:
        obj: A dataclass instance.

    Returns:
        Dict with all fields serialised to plain Python types.

    Raises:
        ValidationError: If *obj* is not a dataclass instance.

    Examples:
        >>> @dataclass
        ... class Foo:
        ...     name: str
        ...     value: int
        >>> dataclass_to_dict(Foo(name="bar", value=42))
        {'name': 'bar', 'value': 42}
    """
    if not is_dataclass(obj) or isinstance(obj, type):
        raise ValidationError(f"Expected dataclass instance, got {type(obj)}")
    result = {}
    for f in fields(obj):
        val = getattr(obj, f.name)
        if isinstance(val, Enum):
            result[f.name] = val.value
        elif isinstance(val, datetime):
            result[f.name] = val.isoformat()
        elif is_dataclass(val) and not isinstance(val, type):
            result[f.name] = dataclass_to_dict(val)
        else:
            result[f.name] = val
    return result
