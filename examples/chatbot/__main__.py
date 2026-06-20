"""Entry point for ``python -m examples.chatbot``."""

from __future__ import annotations

import contextlib

from examples.chatbot.server import main

if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
