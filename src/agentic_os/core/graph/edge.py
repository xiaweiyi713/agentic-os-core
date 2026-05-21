"""Knowledge graph directed weighted edges."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agentic_os.exceptions import ValidationError

logger = logging.getLogger(__name__)


class EdgeType(Enum):
    """Semantic relationship types for directed edges.

    Each type captures a distinct kind of causal or associative link:
        CAUSAL: A caused B.
        TEMPORAL: A occurred before B.
        ASSOCIATIVE: A is related to B.
        DERIVED_FROM: B was inferred from A.
        SUPPORTS: A provides evidence for B.
        CONTRADICTS: A contradicts B.
    """

    CAUSAL = "causal"                # Causal: A caused B
    TEMPORAL = "temporal"            # Temporal: A before B
    ASSOCIATIVE = "associative"      # Associative: A related to B
    DERIVED_FROM = "derived_from"    # Derivation: B derived from A
    SUPPORTS = "supports"            # Support: A supports B
    CONTRADICTS = "contradicts"      # Contradiction: A contradicts B


@dataclass
class Edge:
    """A directed, weighted edge in the knowledge graph.

    Edges represent typed relationships between two nodes with a weight
    in [0, 1] indicating relationship strength.

    Attributes:
        source_id: ID of the origin node.
        target_id: ID of the destination node.
        type: Semantic relationship type.
        weight: Strength of the relationship in [0, 1]. Defaults to 1.0.
        metadata: Arbitrary key-value pairs for extensions.

    Raises:
        ValidationError: If ``weight`` is not in the range [0, 1].

    Example::

        e = Edge(source_id="n1", target_id="n2", type=EdgeType.CAUSAL, weight=0.8)
        assert e.key == ("n1", "n2")
    """

    source_id: str
    target_id: str
    type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Edge({self.source_id[:8]}..-{self.type.value}->{self.target_id[:8]}.., w={self.weight:.2f})"

    def __post_init__(self) -> None:
        logger.debug("edge created %s -> %s type=%s", self.source_id, self.target_id, self.type.value)
        if not 0.0 <= self.weight <= 1.0:
            raise ValidationError(f"Edge weight must be in [0, 1], got {self.weight}")

    @property
    def key(self) -> tuple[str, str]:
        """Return the ``(source_id, target_id)`` tuple for use as a dict key.

        Returns:
            A 2-tuple identifying this edge's endpoints.
        """
        return (self.source_id, self.target_id)
