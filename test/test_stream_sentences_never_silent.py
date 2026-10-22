"""``stream_sentences`` must not drop the end of an answer.

These tests need no model. The engine is built with ``__new__``, and only
``config`` and ``stream_tokens`` are supplied, so they exercise the sentence
logic and nothing else.

``SentenceBoundaryDetector`` reports a boundary only when another sentence
starts after it: ``SENTENCE_BOUNDARY_RE`` carries a lookahead for whitespace
plus an upper-case letter. The last sentence of every answer therefore stays
in ``finish()``, complete or not. Dropping it unread loses the end of the
answer, and for a one-sentence answer the engine says nothing at all while
``continue_chat`` returns the same text.
"""
import pytest

from ovos_gguf_plugin.chat import GGUFChatEngine


def _engine(reply, config=None):
    """A GGUFChatEngine whose token stream is a fixed reply, and no model."""
    engine = GGUFChatEngine.__new__(GGUFChatEngine)
    engine.config = config or {}
    engine.stream_tokens = lambda *args, **kwargs: (
        reply[i:i + 3] for i in range(0, len(reply), 3))
    return engine


@pytest.mark.parametrize("reply", ["Hello.", "Hi there!", "Bonjour !"])
def test_a_one_sentence_answer_is_spoken(reply):
    """The case the e2e test hits: a short complete answer, and the engine
    used to yield nothing for it."""
    assert list(_engine(reply).stream_sentences([])) == [reply]


@pytest.mark.parametrize("reply,expected", [
    ("Hello! How are you?", ["Hello!", "How are you?"]),
    ("Sure. Hello to you as well, friend.",
     ["Sure.", "Hello to you as well, friend."]),
])
def test_the_last_sentence_is_not_dropped(reply, expected):
    assert list(_engine(reply).stream_sentences([])) == expected


def test_a_cut_off_tail_after_speech_is_dropped():
    """The flag keeps its meaning. This tail ends in no terminator, so
    max_tokens cut it, and a sentence was already spoken."""
    reply = "This is a complete sentence. And this tail was cut off"
    assert list(_engine(reply).stream_sentences([])) == [
        "This is a complete sentence."]


def test_a_cut_off_answer_alone_is_still_spoken():
    """Nothing was spoken before it, so silence is the worse answer."""
    reply = "Hello there, I was cut off mid"
    assert list(_engine(reply).stream_sentences([])) == [reply]


def test_the_flag_off_keeps_every_fragment():
    reply = "This is a complete sentence. And this tail was cut off"
    engine = _engine(reply, {"drop_incomplete_sentences": False})
    assert list(engine.stream_sentences([])) == [
        "This is a complete sentence.", "And this tail was cut off"]
