# Localized prompts

System prompts and user-message templates are **not** hardcoded. They are
`.prompt` resource files loaded through
[ovos-spec-tools](https://github.com/OpenVoiceOS/ovos-spec-tools)
(OVOS-INTENT-2 §4.4), so they can be localized.

## Layout

```
ovos_gguf_plugin/locale/
└── en-us/
    ├── translate_system.prompt
    ├── translate_with_source.prompt
    ├── translate_no_source.prompt
    ├── detect_system.prompt
    ├── detect_user.prompt
    ├── summarize_system.prompt
    ├── summarize_user.prompt
    └── dialog_transform_system.prompt
```

A `.prompt` is whole-file plain text. The only special construct is `{name}`
slot substitution, applied **conservatively**: a slot is filled only when the
caller supplies a well-formed name, it is never touched inside fenced code
blocks, and an unfilled `{slot}` is left literal. This makes prompts that embed
JSON or code safe.

Slots in the shipped prompts:

| Prompt | Slots |
|---|---|
| `translate_with_source` | `{source}` `{target}` `{text}` |
| `translate_no_source` | `{target}` `{text}` |
| `detect_user` | `{text}` |
| `summarize_user` | `{content}` |

## Adding a language

Create `ovos_gguf_plugin/locale/<lang>/` (for example `pt-pt/`) and drop translated
`.prompt` files with the same base names. ovos-spec-tools picks them up automatically
and applies its language fallback. The active language is the assistant's configured
`lang`. If a prompt is missing for that language, the plugin falls back to the
`en-us` prompt, so generation never breaks.

A `system_prompt` set in plugin config still overrides the localized one.

---
[← Configuration](configuration.md) · [Home](../README.md) · [Recommended models →](models.md)
