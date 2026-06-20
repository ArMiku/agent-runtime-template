"""A streaming Agent chatbot: tool-based arithmetic + Bilibili video search over MCP.

Run it::

    uv run --env-file .env python -m examples.chatbot

Then open http://127.0.0.1:8000 in a browser. The frontend renders the agent's
thinking process: reasoning deltas, every tool call (name + arguments) and its
result, interleaved with the final answer.
"""
