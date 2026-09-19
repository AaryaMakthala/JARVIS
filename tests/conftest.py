"""Shared pytest setup for the JARVIS test suite.

``langgraph.checkpoint.serde._msgpack`` reads ``LANGGRAPH_STRICT_MSGPACK`` at
module import time, BEFORE any test module loads langgraph (pytest imports this
file first).  Closing that env-var gap is exactly why ``graph.py`` now passes an
explicit secure serializer too -- see ``tests/unit/test_serializer.py``.
"""

from __future__ import annotations

import os

os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"
