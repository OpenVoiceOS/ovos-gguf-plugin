"""Localized prompt loading for the GGUF wrappers (OVOS-INTENT-2 §4.4 ``.prompt``).

Every system prompt and user-message template lives as a
``locale/<lang>/<name>.prompt`` resource and is loaded through
[ovos-spec-tools](https://github.com/OpenVoiceOS/ovos-spec-tools), so the prompts
are localizable: drop a translated ``.prompt`` under a new ``locale/<lang>/`` and
it is picked up automatically (with the spec's smart language fallback). English
(``en-us``) ships by default.

Slot substitution follows §4.4 — ``{name}`` is filled only for supplied,
well-formed slot names and never inside fenced code blocks, so prompts that embed
JSON or code are safe.
"""
import os
from typing import Dict, Optional

from ovos_config import Configuration
from ovos_spec_tools import standardize_lang
from ovos_spec_tools.prompt import render_prompt
from ovos_spec_tools.resources import LocaleResources

LOCALE_DIR = os.path.join(os.path.dirname(__file__), "locale")
DEFAULT_LANG = "en-us"

_RESOURCES = LocaleResources(skill_locale=LOCALE_DIR)


def default_lang() -> str:
    """The assistant's configured language (standardized); ``en-us`` if unset."""
    try:
        return standardize_lang(Configuration().get("lang") or DEFAULT_LANG)
    except Exception:
        return DEFAULT_LANG


def load_prompt(name: str, lang: Optional[str] = None,
                slots: Optional[Dict[str, object]] = None) -> str:
    """Load the localized ``.prompt`` ``name`` for ``lang`` and fill ``slots``.

    ``lang`` defaults to the assistant language. It is resolved against the
    shipped locale dirs with smart fallback; if the prompt still cannot be found
    for it, the default ``en-us`` prompt is used so a missing localization never
    breaks generation. Slots are substituted conservatively (§4.4).

    Args:
        name: base name of the ``.prompt`` (without extension).
        lang: BCP-47 tag to render in; defaults to :func:`default_lang`.
        slots: values for ``{name}`` substitution points.

    Returns:
        The rendered prompt text.
    """
    lang = lang or default_lang()
    try:
        text = _RESOURCES.load_prompt(name, lang)
    except FileNotFoundError:
        text = _RESOURCES.load_prompt(name, DEFAULT_LANG)
    return render_prompt(text, slots or {})
