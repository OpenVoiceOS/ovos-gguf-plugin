"""Tests for localized .prompt loading via ovos-spec-tools.

These cover only the prompt layer (no llama.cpp): the shipped en-us resources
exist, slots fill conservatively, and an unsupported language falls back to en-us.
"""
import os

from ovos_gguf_plugin.prompts import load_prompt, default_lang, LOCALE_DIR

EXPECTED = [
    "translate_system", "translate_with_source", "translate_no_source",
    "detect_system", "detect_user",
    "summarize_system", "summarize_user",
    "dialog_transform_system",
]


def test_all_prompts_ship_for_en_us():
    d = os.path.join(LOCALE_DIR, "en-us")
    for name in EXPECTED:
        path = os.path.join(d, f"{name}.prompt")
        assert os.path.isfile(path), f"missing {path}"
        assert os.path.getsize(path) > 0


def test_translate_system_is_verbatim():
    assert load_prompt("translate_system", "en-us") == \
        "You are a professional translator. Your task is to translate text"


def test_translate_with_source_fills_slots():
    out = load_prompt("translate_with_source", "en-us",
                      {"source": "English", "target": "Spanish", "text": "hello"})
    assert out == "Translate the following text from English into Spanish.\nEnglish: hello\nSpanish: "


def test_unfilled_slot_stays_literal():
    # text not supplied -> {text} stays literal (conservative §4.4)
    out = load_prompt("translate_no_source", "en-us", {"target": "Spanish"})
    assert "{text}" in out
    assert "into Spanish" in out


def test_summarize_user_preserves_braces_in_content():
    # content with JSON braces must survive (no .format-style corruption)
    out = load_prompt("summarize_user", "en-us", {"content": '{"k": 1}'})
    assert '{"k": 1}' in out
    assert "summarize" in out.lower()


def test_unsupported_lang_falls_back_to_en_us():
    # a far language with no locale dir resolves to the shipped en-us prompt
    out = load_prompt("detect_system", "zh-cn")
    assert out == load_prompt("detect_system", "en-us")
    assert "language code" in out


def test_default_lang_returns_a_tag():
    assert isinstance(default_lang(), str) and default_lang()
