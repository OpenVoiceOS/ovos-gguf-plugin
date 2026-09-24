"""Real-model end-to-end tests.

These tests download tiny GGUF models and run actual inference.  They are
included in the standard ``test/`` path so CI picks them up via the
``build_tests.yml`` workflow (``test_path: "test"``).  First run downloads
models to the HuggingFace cache; subsequent runs are fast.

Chat model  : afrideva/Smol-Llama-101M-Chat-v1-GGUF  (~45 MB q2_k)
Embed model : leliuga/all-MiniLM-L6-v2-GGUF           (~23 MB Q4_K_M)
"""
import time

import numpy as np
import pytest

from ovos_gguf_plugin.chat import GGUFChatEngine
from ovos_gguf_plugin.embeddings import GGUFEmbeddings
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

from conftest import hf_retry


# ---------------------------------------------------------------------------
# Chat engine
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chat_engine():
    # The engine downloads the GGUF from the hub on construction; retry on 429.
    return hf_retry(lambda: GGUFChatEngine({
        "model": "afrideva/Smol-Llama-101M-Chat-v1-GGUF",
        "remote_filename": "*q2_k.gguf",
        "max_tokens": 24,
        "verbose": False,
    }))


def test_chat_continue_returns_nonempty_string(chat_engine):
    msgs = [AgentMessage(role=MessageRole.USER, content="Say hello.")]
    t0 = time.time()
    resp = chat_engine.continue_chat(msgs)
    elapsed = time.time() - t0
    assert isinstance(resp.content, str)
    assert len(resp.content.strip()) > 0
    assert resp.role == MessageRole.ASSISTANT
    # sanity: tiny model + 24 tokens should finish well under 30 s
    assert elapsed < 30, f"inference took {elapsed:.1f}s"


def test_chat_stream_sentences_yields_at_least_one_chunk(chat_engine):
    msgs = [AgentMessage(role=MessageRole.USER, content="Say hello.")]
    chunks = list(chat_engine.stream_sentences(msgs))
    assert len(chunks) >= 1
    assert all(isinstance(c, str) and c for c in chunks)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder():
    # The hub download happens during construction; a transient 429 is
    # re-raised by the plugin, so retry it under the backoff guard.
    return hf_retry(lambda: GGUFEmbeddings({"model": "all-MiniLM-L6-v2"}))


def test_embeddings_returns_numpy_vector(embedder):
    vec = embedder.get_embeddings("hello")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert len(vec) > 0


def test_embeddings_vector_length_is_384(embedder):
    # all-MiniLM-L6-v2 produces 384-dim vectors
    vec = embedder.get_embeddings("the quick brown fox")
    assert len(vec) == 384


def test_embeddings_different_texts_differ(embedder):
    v1 = embedder.get_embeddings("cat")
    v2 = embedder.get_embeddings("quantum physics")
    # cosine distance should be non-trivial
    assert not np.allclose(v1, v2)
