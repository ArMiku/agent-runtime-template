"""A live web-search tool exposed to the agent as ``web_search``.

The handler calls Tavily's async search API (``AsyncTavilyClient``) and renders
the top results — title, URL, and Tavily's content snippet for each source,
returned untruncated — so the model can ground fresh, real-world information
(news, prices, recent events, docs) instead of guessing from training data.

The Tavily client and its dependency are loaded lazily, so a missing
``tavily-python`` install or an unset ``TAVILY_API_KEY`` degrades gracefully:
:func:`build_web_search_tool` returns ``None`` and the rest of the chatbot —
arithmetic, Bilibili search, planning — keeps working unchanged.
"""

from __future__ import annotations

import os
from typing import Any

from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.tool import FunctionTool

# Tavily requires max_results in [0, 20].
_MIN_RESULTS = 0
_MAX_RESULTS = 20
_DEFAULT_RESULTS = 5
_VALID_TOPICS = ("general", "news", "finance")


def _render_results(response: dict[str, Any]) -> str:
    """Render a Tavily search response into text for the model.

    Args:
        response: The dict returned by ``AsyncTavilyClient.search``.

    Returns:
        The ranked sources as a numbered list of title / URL / content, or a
        ``No results found.`` line.
    """
    results = response.get("results") or []
    if not results:
        return "No results found."

    lines: list[str] = ["Sources:"]
    for i, item in enumerate(results, start=1):
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        # `content` is Tavily's AI-extracted snippet for the source; pass it
        # through verbatim (no truncation) so the model sees all of it.
        content = str(item.get("content") or "").strip()
        # `published_date` only appears for topic="news"; show it when present.
        published = str(item.get("published_date") or "").strip()
        date_suffix = f" ({published})" if published else ""
        lines.append(f"{i}. {title}{date_suffix}")
        if url:
            lines.append(f"   {url}")
        if content:
            lines.append(f"   {content}")
    return "\n".join(lines)


def build_web_search_tool() -> FunctionTool | None:
    """Build the ``web_search`` function tool backed by Tavily.

    Returns:
        A :class:`FunctionTool` the agent can call for live web search, or
        ``None`` if ``tavily-python`` is not importable or ``TAVILY_API_KEY`` is
        unset — in which case the caller simply omits the tool and the rest of
        the chatbot keeps working.
    """
    # Lazy import: a missing `tavily-python` must not break the whole demo —
    # arithmetic / Bilibili / planning should all still run.
    try:
        from tavily import AsyncTavilyClient
    except ImportError:
        print(
            "[chatbot] WARNING: tavily-python is not installed; web search will be "
            "unavailable. Install it (`uv add tavily-python`) to enable."
        )
        return None

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print(
            "[chatbot] WARNING: TAVILY_API_KEY is not set; web search will be "
            "unavailable. Add it to .env to enable."
        )
        return None

    # One shared async client for the process — cheap to hold, opens its HTTP
    # session lazily on the first search call.
    client = AsyncTavilyClient(api_key)

    async def _web_search(context: ContextWrapper, **kwargs: object) -> str:
        """Search the live web via Tavily and return ranked sources as text.

        Args:
            context: The run context (unused; required by the tool contract).
            **kwargs: Expects ``query`` (required), and optionally ``max_results``,
                ``topic`` ("general" | "news" | "finance"), and ``time_range``
                ("day" | "week" | "month" | "year").

        Returns:
            A numbered list of sources (title / URL / content snippet), or an
            ``error: ...`` message the model can read and recover from.
        """
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return "error: 'query' is required."

        try:
            max_results = int(kwargs.get("max_results", _DEFAULT_RESULTS))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            max_results = _DEFAULT_RESULTS
        # Clamp into Tavily's valid range instead of failing the call.
        max_results = max(_MIN_RESULTS, min(_MAX_RESULTS, max_results))

        topic = str(kwargs.get("topic") or "general").strip() or "general"
        if topic not in _VALID_TOPICS:
            topic = "general"

        time_range = str(kwargs.get("time_range") or "").strip() or None

        try:
            response = await client.search(
                query=query,
                search_depth="basic",
                topic=topic,
                max_results=max_results,
                time_range=time_range,
            )
        except Exception as exc:  # noqa: BLE001 - surface API/network errors to the model
            return f"error: web search failed: {type(exc).__name__}: {exc}"

        # tavily-python returns a plain dict; normalize defensively in case a
        # future version wraps it, then render.
        result = response if isinstance(response, dict) else dict(response)
        return _render_results(result)

    return FunctionTool(
        name="web_search",
        description=(
            "Search the live web for up-to-date information and return the top sources "
            "(title, URL, content snippet) plus a short generated answer. Use this for "
            "anything current or factual you don't already know — news, recent events, "
            "docs, prices, people — instead of guessing."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g. 'latest Claude model release date'.",
                },
                "max_results": {
                    "type": "number",
                    "description": "How many sources to return (0–20). Defaults to 5.",
                },
                "topic": {
                    "type": "string",
                    "enum": ["general", "news", "finance"],
                    "description": "Search category. Use 'news' for recent events, 'finance' for markets. Defaults to 'general'.",
                },
                "time_range": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year"],
                    "description": "Restrict to results within a recency window. Omit for no limit.",
                },
            },
            "required": ["query"],
        },
        handler=_web_search,
    )
