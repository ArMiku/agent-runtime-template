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

__all__ = ["logger", "configure_logging", "LogPipe"]

#: Module-level logger consumed as ``logger.info(...)`` throughout the runtime.
#:
#: ``logging.Logger`` exposes ``warning/debug/error/info/exception`` natively; ``warn``
#: is a (deprecated) stdlib alias that some call sites use, so we normalise it here.
logger: logging.Logger = logging.getLogger("agent_runtime")


def configure_logging(level: int | str = logging.INFO) -> None:
    """Attach a best-effort handler + level to :data:`logger`.

    Idempotent: safe to call multiple times. Consumers wiring the runtime into a larger
    application may instead configure the ``agent_runtime`` logger through their own
    logging setup and ignore this helper entirely.
    """
    logger.setLevel(level)
    if not any(getattr(h, "_agent_runtime_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
        handler._agent_runtime_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False


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
