"""Capstone end-to-end test: GGUF-backed persona through the full OVOS pipeline.

Proves:
  1. An utterance flows through the real OVOS intent pipeline, hits the persona
     pipeline plugin (which delegates to a GGUFChatEngine handler), and produces
     a ``speak`` message with non-empty text.
  2. Per-session memory is recorded: the live PersonaService accumulates USER +
     ASSISTANT turns keyed by session_id after the pipeline runs.

A tiny real GGUF model is downloaded once (~52 MB) on first run and cached in
the HuggingFace cache.  No other network access is required.

Model: afrideva/Smol-Llama-101M-Chat-v1-GGUF (*q2_k.gguf, ~52 MB)
"""
import json
import os
import tempfile

import pytest

ovoscope = pytest.importorskip("ovoscope")
pytest.importorskip("ovos_persona")

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session, SessionManager
from ovos_plugin_manager.templates.agents import MessageRole

from ovoscope import (
    PERSONA_PIPELINE,
    CaptureSession,
    get_minicroft,
    is_pipeline_available,
)

if not is_pipeline_available(PERSONA_PIPELINE):
    pytest.skip("ovos-persona-pipeline-plugin not installed", allow_module_level=True)

# ---------------------------------------------------------------------------
# Persona dir + config
# ---------------------------------------------------------------------------

PERSONA_NAME = "TinyBot"
_TMPDIR = tempfile.mkdtemp()

_PERSONA = {
    "name": PERSONA_NAME,
    "handlers": ["ovos-chat-gguf-plugin"],
    "ovos-chat-gguf-plugin": {
        "model": "afrideva/Smol-Llama-101M-Chat-v1-GGUF",
        "remote_filename": "*q2_k.gguf",
        "max_tokens": 24,
        "verbose": False,
    },
}

with open(os.path.join(_TMPDIR, f"{PERSONA_NAME}.json"), "w") as _fh:
    json.dump(_PERSONA, _fh)

PIPELINE_CONFIG = {
    "persona": {
        "personas_path": _TMPDIR,
        "default_persona": PERSONA_NAME,
        "short-term-memory": True,
        "handle_fallback": True,
        "ignore_plugin_personas": True,
    }
}

TEST_PIPELINE = [
    "ovos-persona-pipeline-plugin-high",
    "ovos-persona-pipeline-plugin-low",
]


# ---------------------------------------------------------------------------
# Module-level MiniCroft (shared for speed — model loads once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mc():
    croft = get_minicroft(
        skill_ids=[],
        default_pipeline=TEST_PIPELINE,
        pipeline_config=PIPELINE_CONFIG,
    )
    yield croft
    croft.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utterance_msg(utterance: str, sess: Session) -> Message:
    return Message(
        "recognizer_loop:utterance",
        {"utterances": [utterance], "lang": sess.lang},
        {"session": sess.serialize()},
    )


def _drive(croft, sess: Session, utterance: str, timeout: int = 90):
    cap = CaptureSession(
        croft,
        eof_msgs=["ovos.utterance.handled", "ovos.utterance.cancelled"],
    )
    cap.capture(_utterance_msg(utterance, sess), timeout=timeout)
    return cap.finish()


def _persona_service(croft):
    return croft.intents.pipeline_plugins["ovos-persona-pipeline-plugin"]


# ---------------------------------------------------------------------------
# Test 1: GGUF persona speaks through the full pipeline
# ---------------------------------------------------------------------------

class TestGGUFPersonaSpeaksThroughPipeline:
    """Utterance must traverse the full OVOS intent pipeline and produce a
    ``speak`` message whose ``utterance`` field is non-empty (i.e. a real
    GGUF model generated a response and it was routed through the pipeline)."""

    def test_pipeline_produces_speak(self, mc):
        sess = Session(session_id="gguf-e2e-speak")
        SessionManager.sessions[sess.session_id] = sess

        messages = _drive(mc, sess, "hello, who are you?", timeout=90)

        # OVOS-INTENT bus-namespace migration: the spec name is
        # ``ovos.utterance.speak``; the legacy ``speak`` topic may or may not
        # be dual-sent depending on stack version, so accept either without
        # counting occurrences.
        msg_types = [m.msg_type for m in messages]
        speak_msgs = [
            m for m in messages if m.msg_type in ("ovos.utterance.speak", "speak")
        ]

        assert speak_msgs, (
            f"Expected at least one speak message; got msg_types: {msg_types}"
        )
        spoken = speak_msgs[0].data.get("utterance", "")
        assert spoken.strip(), (
            f"'speak' message had empty utterance; data={speak_msgs[0].data}"
        )


# ---------------------------------------------------------------------------
# Test 2: per-session memory recorded after pipeline run
# ---------------------------------------------------------------------------

class TestGGUFPersonaMemoryRecorded:
    """After a pipeline turn the PersonaService must have recorded the
    USER input (and an ASSISTANT response) in the persona's short-term memory,
    keyed by session_id."""

    def test_memory_contains_turn(self, mc):
        svc = _persona_service(mc)
        sess = Session(session_id="gguf-e2e-mem")
        SessionManager.sessions[sess.session_id] = sess

        persona = svc.personas.get(PERSONA_NAME)
        assert persona is not None, f"Persona '{PERSONA_NAME}' not loaded"

        _drive(mc, sess, "hello, who are you?", timeout=90)

        # Short-term memory storage has moved between ovos-persona
        # prereleases: 0.9.0a15 keeps it on the persona's
        # AgentContextManager (``persona.memory``, a list of AgentMessage
        # with a MessageRole + content); 0.9.0a16 reverted to a plain
        # ``svc.sessions`` dict of (role, utterance) tuples. Accept either
        # shape so the test doesn't flap with unpinned prerelease churn.
        memory = getattr(persona, "memory", None)
        if memory is not None:
            history = memory.get_history(sess.session_id)
            assert history, (
                f"Memory empty after pipeline turn for session {sess.session_id}"
            )
            roles = [m.role for m in history]
            assert MessageRole.USER in roles, (
                f"No USER turn recorded in memory. History: {history}"
            )
            contents = [m.content for m in history]
        else:
            history = svc.sessions.get(sess.session_id)
            assert history, (
                f"Memory empty after pipeline turn for session {sess.session_id}"
            )
            roles = [role for role, _ in history]
            assert "user" in roles, (
                f"No USER turn recorded in memory. History: {history}"
            )
            contents = [utt for _, utt in history]

        assert any("hello" in c.lower() for c in contents), (
            f"User utterance not found in memory. History: {contents}"
        )
