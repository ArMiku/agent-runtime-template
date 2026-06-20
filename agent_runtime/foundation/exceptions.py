"""Runtime exceptions.

Only :class:`EmptyModelOutputError` is needed by the runtime (the runner's empty-output
retry guard), so this module stays minimal rather than defining a broad error hierarchy.
"""

from __future__ import annotations

__all__ = ["EmptyModelOutputError"]


class EmptyModelOutputError(Exception):
    """Raised when the model response contains no usable assistant output."""
