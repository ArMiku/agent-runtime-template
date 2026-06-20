"""Logging shim for the agent runtime package.

The portable runtime consumes a module-level ``logger`` object as
``logger.info/.debug/.warning/.error/.exception``. Only standard ``logging.Logger``
method names are used (no loguru-specific ``.bind/.opt/.catch/.trace`` calls), so a plain
stdlib ``logging.Logger`` is a faithful, dependency-free implementation.

This module also provides :class:`LogPipe`, a standalone thread that pipes a subprocess'
stdout/stderr into a ``logging.Logger``; the MCP client uses it to capture MCP server
subprocess output.
"""

from __future__ import annotations

import logging
import os
import threading
from logging import Logger
from logging.handlers import RotatingFileHandler

__all__ = ["logger", "configure_logging", "quiet_noisy_loggers", "LogPipe"]

#: Module-level logger consumed as ``logger.info(...)`` throughout the runtime.
#:
#: ``logging.Logger`` exposes ``warning/debug/error/info/exception`` natively; ``warn``
#: is a (deprecated) stdlib alias that some call sites use, so we normalise it here.
logger: logging.Logger = logging.getLogger("agent_runtime")

# Library etiquette (Python logging HOWTO): attach a NullHandler to our top-level
# logger at import time. This is the *only* permitted import-time global side
# effect — it does nothing on its own, so it neither configures logging on the
# host's behalf nor disables propagation. Without it, a host that has not
# configured logging would see the stdlib "No handlers could be found for logger
# 'agent_runtime'" warning (and records could leak to the root ``lastResort``
# handler). ``propagate`` is intentionally left at its default ``True`` so that a
# host still receives our records.
logger.addHandler(logging.NullHandler())


#: Third-party loggers whose default chattiness drowns out the runtime's own
#: records. Quieted to WARNING, but only from inside :func:`configure_logging`
#: (a library MUST NOT mutate other loggers' levels at import time).
_NOISY_LOGGERS: tuple[str, ...] = ("mcp", "httpx", "openai", "anthropic", "aiohttp")

#: Console/file format carrying source location (filename:lineno) for parity
#: with the host's ``[file:line]`` — plain stdlib style, no color.
_STANDALONE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(filename)s:%(lineno)d | %(message)s"


def quiet_noisy_loggers(level: int = logging.WARNING) -> None:
    """Lower the level of chatty third-party loggers (see :data:`_NOISY_LOGGERS`).

    Call this only from a standalone entry point. When ``agent_runtime`` is hosted
    by a larger application, the host owns this decision.
    """
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)


def configure_logging(
    level: int | str | None = None,
    *,
    log_file: str | os.PathLike[str] | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    """Attach console (and optional rotating-file) handlers to :data:`logger`.

    This is a **standalone-only** convenience for running the runtime on its own
    (scripts, examples, research). When ``agent_runtime`` is embedded in a larger
    application, the host configures the ``agent_runtime`` logger
    through its own logging setup and need not call this at all.

    Idempotent: safe to call repeatedly without stacking duplicate handlers.

    Args:
        level: Log level. When ``None``, falls back to the ``AGENT_RUNTIME_LOG_LEVEL``
            environment variable, then to ``INFO``.
        log_file: When set, also write to this path via a ``RotatingFileHandler``.
            Parent directories are created automatically.
        max_bytes: Rotation threshold for the file handler.
        backup_count: Number of rotated backups to retain.
    """
    if level is None:
        level = os.environ.get("AGENT_RUNTIME_LOG_LEVEL", logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(_STANDALONE_FORMAT)

    if not any(getattr(h, "_agent_runtime_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler._agent_runtime_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)

    if log_file is not None and not any(getattr(h, "_agent_runtime_file_handler", False) for h in logger.handlers):
        os.makedirs(os.path.dirname(os.fspath(log_file)) or ".", exist_ok=True)
        file_handler = RotatingFileHandler(
            os.fspath(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler._agent_runtime_file_handler = True  # type: ignore[attr-defined]
        logger.addHandler(file_handler)

    # Standalone: we own the output, so silence the stdlib lastResort path and
    # avoid double printing through the root logger.
    logger.propagate = False
    quiet_noisy_loggers()


class LogPipe(threading.Thread):
    """Pipe a file descriptor (typically a subprocess' stdout/stderr) into a logger.

    Depends only on a stdlib ``logging.Logger`` and ``os.pipe``, so it is fully portable.
    """

    def __init__(
        self,
        level,
        logger: Logger,
        identifier=None,
        callback=None,
    ) -> None:
        threading.Thread.__init__(self)
        self.daemon = True
        self.level = level
        self.fd_read, self.fd_write = os.pipe()
        self.identifier = identifier
        self.logger = logger
        self.callback = callback
        self.reader = os.fdopen(self.fd_read)
        self.start()

    def fileno(self):
        return self.fd_write

    def run(self) -> None:
        for line in iter(self.reader.readline, ""):
            if self.callback:
                self.callback(line.strip())
            self.logger.log(self.level, f"[{self.identifier}] {line.strip()}")

        self.reader.close()

    def close(self) -> None:
        os.close(self.fd_write)
