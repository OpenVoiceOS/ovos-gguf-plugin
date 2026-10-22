"""Regression test for the ``tools`` kwarg contract.

``ovos_plugin_manager.templates.agents.ChatEngine.continue_chat`` declares
``tools`` unconditionally in its base signature. Callers such as
``ovos_persona_server/server_tools.py`` (and the ovos-agentic-loop ReAct
fallback) pass ``tools=`` by keyword to *any* configured ChatEngine,
including non-tool-capable ones like ``GGUFChatEngine``. Python validates
the call signature before the function body runs, so omitting the
parameter raises ``TypeError`` regardless of what the engine would have
done with it.

The backend (``llama_cpp.Llama``) is stubbed out entirely — no network,
no model download.
"""
from unittest.mock import MagicMock

import pytest

from ovos_gguf_plugin.chat import GGUFChatEngine
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole


def _make_engine() -> GGUFChatEngine:
    fake_llama = MagicMock()
    fake_llama.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "stubbed response"}}]
    }
    return GGUFChatEngine(config={}, gguf_engine=fake_llama)


def _messages():
    return [AgentMessage(role=MessageRole.USER, content="hello")]


def test_continue_chat_accepts_tools_none():
    engine = _make_engine()
    result = engine.continue_chat(_messages(), session_id="default",
                                   lang=None, units=None, tools=None)
    assert isinstance(result, AgentMessage)
    assert result.role == MessageRole.ASSISTANT
    assert result.content == "stubbed response"


def test_continue_chat_accepts_tools_list():
    engine = _make_engine()
    fake_tools = [{"type": "function", "function": {"name": "noop"}}]
    result = engine.continue_chat(_messages(), session_id="default",
                                   lang=None, units=None, tools=fake_tools)
    assert isinstance(result, AgentMessage)
    assert result.content == "stubbed response"


def test_continue_chat_accepts_tools_as_keyword_only_call():
    """Mirrors ovos_persona_server/server_tools.py's call shape: everything
    after ``messages`` passed by keyword, including ``tools``."""
    engine = _make_engine()
    result = engine.continue_chat(
        messages=_messages(),
        session_id="default",
        lang=None,
        units=None,
        tools=[{"type": "function", "function": {"name": "noop"}}],
    )
    assert isinstance(result, AgentMessage)
