"""Package importability and version sanity checks."""

from __future__ import annotations

import importlib

import pytest


def test_version_is_string() -> None:
    from jarvis import __version__

    assert isinstance(__version__, str)
    assert __version__.startswith(("0.1.0", "0."))


def test_version_installed_distribution() -> None:
    from importlib import metadata

    from jarvis import __version__

    dist = metadata.version("jarvis-agent")
    assert __version__ == dist


@pytest.mark.parametrize(
    "module",
    [
        "jarvis",
        "jarvis.config",
        "jarvis.secrets",
        "jarvis.logging_setup",
        "jarvis.platform_guard",
        "jarvis.llm.client",
        "jarvis.llm.prompts",
        "jarvis.agent.state",
        "jarvis.agent.graph",
        "jarvis.agent.runner",
        "jarvis.policy.tiers",
        "jarvis.policy.engine",
        "jarvis.policy.paths",
        "jarvis.policy.unlock",
        "jarvis.tools.base",
        "jarvis.tools.registry",
        "jarvis.tools.apps",
        "jarvis.tools.files",
        "jarvis.tools.web",
        "jarvis.tools.keyboard",
        "jarvis.tools.whatsapp",
        "jarvis.tools.dictation",
        "jarvis.tools.system",
        "jarvis.daemon.protocol",
        "jarvis.daemon.server",
        "jarvis.daemon.client",
        "jarvis.daemon.autostart",
        "jarvis.voice.wake",
        "jarvis.voice.stt",
        "jarvis.voice.tts",
        "jarvis.voice.vad",
        "jarvis.memory.db",
        "jarvis.memory.skills",
        "jarvis.memory.failures",
        "jarvis.memory.prefs",
        "jarvis.memory.embeddings",
        "jarvis.audit.checks",
        "jarvis.audit.report",
    ],
)
def test_module_imports_on_any_os(module: str) -> None:
    """Every package module must import without touching Windows-only APIs."""
    importlib.import_module(module)
