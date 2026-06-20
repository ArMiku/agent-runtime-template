"""Session-level plugin enable/disable (spec "会话级插件启停").

A neutral, ``event``-free check: the enabled/disabled lists are read through the
injectable ``PluginStore`` seam, keyed by session. No ``event`` object is involved.
"""

from __future__ import annotations

from .store import PluginStore

__all__ = ["is_plugin_enabled_for_session", "SESSION_CONFIG_PLUGIN_ID", "SESSION_CONFIG_KEY"]

# The session-startup config lives under a reserved plugin_id bucket in the PluginStore.
SESSION_CONFIG_PLUGIN_ID = "__session_plugin_config__"
SESSION_CONFIG_KEY = "config"


async def is_plugin_enabled_for_session(
    store: PluginStore,
    session_id: str,
    plugin_name: str,
) -> bool:
    """Return whether ``plugin_name`` is enabled for ``session_id``.

    Resolution:
    * if the plugin is in the session's ``disabled_plugins`` → ``False``;
    * if in ``enabled_plugins`` → ``True``;
    * otherwise default ``True`` (backward compatible).

    The config shape stored under ``(SESSION_CONFIG_PLUGIN_ID, SESSION_CONFIG_KEY)`` is::

        {"<session_id>": {"enabled_plugins": [...], "disabled_plugins": [...]}}
    """
    all_config = await store.get(SESSION_CONFIG_PLUGIN_ID, SESSION_CONFIG_KEY, {}) or {}
    session_config = all_config.get(session_id, {})

    disabled = session_config.get("disabled_plugins", [])
    enabled = session_config.get("enabled_plugins", [])

    if plugin_name in disabled:
        return False
    if plugin_name in enabled:
        return True
    return True
