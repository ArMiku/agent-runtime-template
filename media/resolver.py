"""Injectable media-resolution seam.

Multimodal input (image / audio references -> provider-deliverable payloads) flows
through a :class:`MediaResolver` injected into the provider. The default implementation,
:class:`DefaultMediaResolver`, is a full media-processing body (download / parse / base64
/ MIME / image compress+transcode / audio magic), with the QQ/WeChat ``silk`` codec
factored out behind an optional
:class:`~agent_runtime.media.media_utils.AudioCodecHook`.

This keeps the runtime free of any hard media-utils dependency while preserving
near-complete media capability out of the box.

The seam resolves a reference into a :class:`~agent_runtime.media.media_utils.ResolvedMediaData`
(rather than a bare string) so callers retain the resolved ``mime_type`` / ``format`` /
``to_data_url()`` they already consume.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .media_utils import (
    AudioCodecHook,
    ResolvedMediaData,
    describe_media_ref,
    get_audio_codec_hook,
    is_file_uri,
    resolve_media_ref_to_base64_data,
    set_audio_codec_hook,
)

__all__ = [
    "MediaResolver",
    "DefaultMediaResolver",
    "default_media_resolver",
    "set_default_media_resolver",
    # Optional silk codec hook (re-exported for convenience).
    "AudioCodecHook",
    "set_audio_codec_hook",
    "get_audio_codec_hook",
]


@runtime_checkable
class MediaResolver(Protocol):
    """Injectable media-resolution seam consumed by provider sources.

    A reference may be a remote URL, a ``file://`` URI, a ``data:`` URI, or a bare
    base64 payload; the resolver normalises it into a
    :class:`~agent_runtime.media.media_utils.ResolvedMediaData` carrying the bytes,
    MIME type, format, and a ``to_data_url()`` helper.
    """

    async def resolve(
        self,
        media_ref: Any,
        *,
        media_type: str,
        strict: bool = False,
    ) -> ResolvedMediaData | None:
        """Resolve ``media_ref`` into structured base64 media data (or ``None``)."""
        ...

    def describe(self, media_ref: Any) -> str:
        """Return a log-safe, human-readable description of ``media_ref``."""
        ...

    @staticmethod
    def is_file_uri(media_ref: Any) -> bool:
        """Return ``True`` if ``media_ref`` is a ``file://`` URI."""
        ...


class DefaultMediaResolver:
    """Default :class:`MediaResolver` backed by the ported ``media_utils`` body."""

    async def resolve(
        self,
        media_ref: Any,
        *,
        media_type: str,
        strict: bool = False,
    ) -> ResolvedMediaData | None:
        return await resolve_media_ref_to_base64_data(
            media_ref,
            media_type=media_type,
            strict=strict,
        )

    def describe(self, media_ref: Any) -> str:
        return describe_media_ref(media_ref)

    @staticmethod
    def is_file_uri(media_ref: Any) -> bool:
        return is_file_uri(media_ref)


#: Process-wide default resolver injected into providers that don't override.
default_media_resolver: MediaResolver = DefaultMediaResolver()


def set_default_media_resolver(resolver: MediaResolver) -> None:
    """Swap the process-wide default media resolver."""
    global default_media_resolver
    default_media_resolver = resolver
