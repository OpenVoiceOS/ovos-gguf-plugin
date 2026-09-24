"""Regression test for https://github.com/OpenVoiceOS/ovos-gguf-plugin/issues/25

``GGUFDialogTransformer`` used to default its ``name`` to the OpenAI plugin's
entry-point id (a copy-paste mistake). Since the ``name`` is used to look up
this plugin's own config section, a wrong default silently reads the wrong
(or no) configuration at runtime. This test pins the default to the plugin's
actual registered entry-point name.
"""
from unittest import mock

from ovos_gguf_plugin.dialog_transformers import GGUFDialogTransformer


def test_dialog_transformer_default_name_matches_entry_point():
    # gguf_engine stubbed out so no real model is downloaded/loaded
    transformer = GGUFDialogTransformer(gguf_engine=mock.MagicMock())
    assert transformer.name == "ovos-dialog-transformer-gguf-plugin"
    assert transformer.name != "ovos-dialog-transformer-openai-plugin"
