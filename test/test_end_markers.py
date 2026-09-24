"""The chat template's end marker must not reach the spoken line.

A small or heavily quantised build emits the template's end string as
literal TEXT instead of as its end-of-sequence token. `smol-llama-101m-chat`
at q2_k does it, and four of six spoken lines on a real bus ended in
`<|im_end|>` (live-ovos-gguf-plugin-38.md).

Nothing downstream removed it. `stream_sentences` feeds the token stream to
`SentenceBoundaryDetector`, the persona pipeline forwards each yielded
string, and `PersonaService.handle_persona_query` speaks it.

The marker also HIDES a sentence boundary, which is why it is stripped from
the token stream rather than from the finished sentence: the detector wants a
terminator followed by a space, and `"...part.<|im_end|>"` gives it neither.
"""
import importlib.util
import sys
import types
from pathlib import Path

CHAT = Path(__file__).resolve().parents[1] / "ovos_gguf_plugin" / "chat.py"


def _module():
    """Load chat.py alone, without llama_cpp or a model."""
    for name, attrs in {
        "llama_cpp": ["Llama"],
        "ovos_plugin_manager": [],
        "ovos_plugin_manager.templates": [],
        "ovos_plugin_manager.templates.agents":
            ["ChatEngine", "AgentMessage", "MessageRole", "ToolsArg",
             "SummarizerEngine"],
        "ovos_utils": [],
        "ovos_utils.log": ["LOG"],
    }.items():
        if name in sys.modules:
            continue
        stub = types.ModuleType(name)
        for attr in attrs:
            setattr(stub, attr, type(attr, (), {}))
        sys.modules[name] = stub
    spec = importlib.util.spec_from_file_location("gguf_chat_under_test", CHAT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stripped(chunks):
    return "".join(_module().strip_end_markers(chunks))


def test_the_observed_stream_loses_its_marker():
    """The shape the live cell spoke: a marker glued to the full stop."""
    assert _stripped(["The first part is complex.", "<|im_end|>"]) == \
        "The first part is complex."


def test_a_marker_split_across_chunks_is_still_removed():
    """A token boundary can fall anywhere, including inside the marker."""
    assert _stripped(["Answer here.", "<|im", "_end|>"]) == "Answer here."
    assert _stripped(["Answer here.", "<", "|im_end", "|>"]) == "Answer here."


def test_a_marker_arriving_one_character_at_a_time_is_removed():
    """The realistic case: llama.cpp streams small pieces."""
    assert _stripped(list("Done.<|im_end|>")) == "Done."


def test_a_marker_in_the_middle_of_a_stream_is_removed():
    assert _stripped(["One.", "<|im_end|>", " Two."]) == "One. Two."


def test_text_that_is_not_a_marker_is_untouched():
    """Nothing is dropped from an ordinary answer, `<` included."""
    assert _stripped(["Hello there. ", "How are you? "]) == \
        "Hello there. How are you? "
    assert _stripped(["a < b is true. "]) == "a < b is true. "


def test_a_partial_marker_at_the_end_of_the_stream_is_kept():
    """Held-back text is yielded when the stream ends.

    A partial marker is only partial once there is nothing more coming, and
    dropping real text would be the worse mistake.
    """
    assert _stripped(["Text ", "<|im"]) == "Text <|im"


def test_every_marker_in_the_default_set_is_removed():
    mod = _module()
    for marker in mod.END_MARKERS:
        assert "".join(mod.strip_end_markers(["Done.", marker])) == "Done."


def test_the_marker_set_is_configurable():
    mod = _module()
    out = "".join(mod.strip_end_markers(["Done.", "<|custom|>"],
                                        markers=("<|custom|>",)))
    assert out == "Done."


def test_the_stripped_stream_gives_the_boundary_back():
    """The point of stripping the TOKENS and not the finished sentence.

    With the marker glued to the full stop, `SentenceBoundaryDetector`
    reports no boundary at all and the whole answer stays in `finish()`,
    where `drop_incomplete_sentences` decides whether it is spoken. Stripped,
    the stream behaves like any other.
    """
    from sentence_stream import SentenceBoundaryDetector

    raw = ["One sentence.", "<|im_end|>", " Another one. "]
    detector = SentenceBoundaryDetector()
    before = []
    for chunk in raw:
        before.extend(detector.add_chunk(chunk))

    detector = SentenceBoundaryDetector()
    after = []
    for chunk in _module().strip_end_markers(raw):
        after.extend(detector.add_chunk(chunk))

    assert before == []
    assert after == ["One sentence."]


class _FakeModel:
    """Yields the chunk shape llama.cpp yields, marker included."""

    def __init__(self, chunks):
        self._chunks = chunks

    def create_chat_completion(self, **kwargs):
        assert kwargs.get("stream") is True
        for chunk in self._chunks:
            yield {"choices": [{"delta": {"content": chunk}}]}


def _engine(chunks, config=None):
    """A GGUFChatEngine that never loads a model.

    `__init__` downloads and opens a gguf file, so the instance is built
    without it and the two attributes the method reads are set directly.
    This drives the REAL `stream_tokens`, which is the point: a test that
    only calls `strip_end_markers` passes even when nothing calls it.
    """
    mod = _module()
    engine = object.__new__(mod.GGUFChatEngine)
    engine.model = _FakeModel(chunks)
    engine.config = config if config is not None else {}
    engine.validate_messages = lambda messages: messages
    return engine


def test_stream_tokens_itself_strips_the_marker():
    engine = _engine(["The first part is complex.", "<|im_end|>"])
    assert "".join(engine.stream_tokens([])) == "The first part is complex."


def test_stream_tokens_honours_a_configured_marker_set():
    engine = _engine(["Done.", "<|custom|>"],
                     config={"end_markers": ["<|custom|>"]})
    assert "".join(engine.stream_tokens([])) == "Done."


def test_empty_markers_list_passes_the_stream_through():
    """An empty marker set must disable stripping, not raise.

    The empty list is the natural way to switch the feature off in a config.
    """
    assert _stripped(["Done.", "<|im_end|>"]) == "Done."
    # With empty markers, everything passes through
    mod = _module()
    result = "".join(mod.strip_end_markers(["Done.", "<|im_end|>"], markers=[]))
    assert result == "Done.<|im_end|>"


def test_plain_string_end_markers_must_be_rejected_or_wrapped():
    """A plain string marker must be rejected or treated as a single marker.

    Users naturally write a single marker as a string in config, not as a list.
    Iterating the string as characters is a bug that silently deletes text.
    """
    mod = _module()
    # This should either raise with a clear error or treat the string as one marker
    try:
        result = "".join(mod.strip_end_markers(["Hello there, friend."], 
                                                markers="<|im_end|>"))
        # If it doesn't raise, it must work correctly
        assert result == "Hello there, friend."
    except (TypeError, ValueError) as e:
        # If it raises, the error should be clear
        assert "string" in str(e).lower() or "marker" in str(e).lower()
