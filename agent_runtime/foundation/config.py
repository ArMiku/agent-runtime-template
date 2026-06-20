"""Application-config seam.

Resolving media for a remote LLM can take two forms: inline data URIs (base64), or a
public callback URL that exposes a locally-downloaded file over HTTP. The latter needs
two host-provided pieces:

* a public base URL of a file-callback service, and
* a way to register a local file and obtain an access token, so a media component can be
  addressed as ``{callback_base}/api/file/{token}``.

These are host-application concerns, not runtime concerns. Rather than carry any singleton
(or a DB-backed token service) into the package, we expose a single injectable seam
:class:`FileCallbackService`. The default implementation is *disabled* —
``get_callback_base()`` returns ``None`` and ``register_file`` returns ``""`` — which makes
media components resolve to data URIs (base64) instead of callback URLs. This is the
correct behaviour for a headless runtime with no public file endpoint.

Consumers that host a callback endpoint inject their own implementation via
:func:`set_file_callback_service`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = [
    "FileCallbackService",
    "DisabledFileCallback",
    "get_file_callback_service",
    "set_file_callback_service",
]


@runtime_checkable
class FileCallbackService(Protocol):
    """Optional seam exposing locally-resolved media over an HTTP callback endpoint."""

    def get_callback_base(self) -> str | None:
        """Return the public base URL for the callback file service, or ``None`` if disabled."""
        ...

    async def register_file(self, file_path: str) -> str:
        """Register a local file and return an opaque access token (``""`` if disabled)."""
        ...


class DisabledFileCallback:
    """Default no-op implementation: callback hosting is disabled."""

    def get_callback_base(self) -> str | None:
        return None

    async def register_file(self, file_path: str) -> str:
        return ""


_service: FileCallbackService = DisabledFileCallback()


def get_file_callback_service() -> FileCallbackService:
    """Return the active file-callback service (disabled by default)."""
    return _service


def set_file_callback_service(service: FileCallbackService) -> None:
    """Inject a file-callback service implementation."""
    global _service
    _service = service
