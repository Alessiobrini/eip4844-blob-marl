"""BuilderAgent: Mesa agent representing the block builder.

The builder is a pass-through that always includes all blob transactions.
A learning builder is outside the scope of this code.
"""

from __future__ import annotations

from typing import Any

import mesa


class BuilderAgent(mesa.Agent):
    """Phase 1 builder: always includes all pending type-3 transactions.

    Args:
        model: Parent Mesa model.
    """

    def __init__(self, model: mesa.Model) -> None:
        super().__init__(model)

    def include(self, pending_tx: list[Any]) -> list[Any]:
        """Include all pending blob transactions (pass-through).

        Args:
            pending_tx: List of pending type-3 transactions.

        Returns:
            The same list, unmodified.
        """
        return pending_tx

    def step(self) -> None:
        """Mesa step hook.  No-op in Phase 1."""
