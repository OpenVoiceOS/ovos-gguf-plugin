"""Hermetic tests for GGUFEmbeddings (no model downloads, no native llama.cpp).

The native ``llama_cpp`` extension and all hub/disk access are mocked, so the
suite exercises only this plugin's model-resolution logic. CI installs the real
``llama-cpp-python`` via ``pip install .[test]`` and validates the import.
"""
import sys
import types
from unittest import mock

import numpy as np

# Stub the native extension so the suite runs without the heavy wheel.
if "llama_cpp" not in sys.modules:
    _stub = types.ModuleType("llama_cpp")
    _stub.Llama = mock.MagicMock(name="Llama")
    sys.modules["llama_cpp"] = _stub

from ovos_gguf_plugin.embeddings import GGUFEmbeddings


def _fake_model(vector=(0.1, 0.2, 0.3)):
    model = mock.MagicMock(name="LlamaModel")
    model.create_embedding.return_value = {"data": [{"embedding": list(vector)}]}
    return model


def test_local_path_loads_and_embeds():
    fake = _fake_model()
    with mock.patch("ovos_gguf_plugin.embeddings.Llama") as MockLlama, \
            mock.patch("ovos_gguf_plugin.embeddings.os.path.isfile", return_value=True):
        MockLlama.return_value = fake
        emb = GGUFEmbeddings(config={"model": "/models/foo.gguf"})

    MockLlama.assert_called_once()
    _, kwargs = MockLlama.call_args
    assert kwargs["model_path"] == "/models/foo.gguf"
    assert kwargs["embedding"] is True
    vec = emb.get_embeddings("hello")
    assert isinstance(vec, np.ndarray)
    assert np.allclose(vec, [0.1, 0.2, 0.3])


def test_default_model_name_resolves_to_hub_repo():
    with mock.patch("ovos_gguf_plugin.embeddings.Llama") as MockLlama, \
            mock.patch("ovos_gguf_plugin.embeddings.os.path.isfile", return_value=False):
        MockLlama.from_pretrained.return_value = _fake_model()
        GGUFEmbeddings(config={"model": "labse"})

    MockLlama.from_pretrained.assert_called_once()
    _, kwargs = MockLlama.from_pretrained.call_args
    assert kwargs["repo_id"] == "ChristianAzinn/labse-gguf"
    assert kwargs["filename"] == "labse.Q4_K_M.gguf"
    assert kwargs["embedding"] is True


def test_default_when_no_model_configured_is_labse():
    with mock.patch("ovos_gguf_plugin.embeddings.Llama") as MockLlama, \
            mock.patch("ovos_gguf_plugin.embeddings.os.path.isfile", return_value=False):
        MockLlama.from_pretrained.return_value = _fake_model()
        GGUFEmbeddings()

    _, kwargs = MockLlama.from_pretrained.call_args
    assert kwargs["repo_id"] == "ChristianAzinn/labse-gguf"


def test_bare_repo_id_uses_remote_filename_glob():
    with mock.patch("ovos_gguf_plugin.embeddings.Llama") as MockLlama, \
            mock.patch("ovos_gguf_plugin.embeddings.os.path.isfile", return_value=False):
        MockLlama.from_pretrained.return_value = _fake_model()
        GGUFEmbeddings(config={"model": "myorg/custom-embed-gguf", "remote_filename": "*q8_0.gguf"})

    _, kwargs = MockLlama.from_pretrained.call_args
    assert kwargs["repo_id"] == "myorg/custom-embed-gguf"
    assert kwargs["filename"] == "*q8_0.gguf"


def test_extra_config_forwarded_model_keys_stripped():
    with mock.patch("ovos_gguf_plugin.embeddings.Llama") as MockLlama, \
            mock.patch("ovos_gguf_plugin.embeddings.os.path.isfile", return_value=False):
        MockLlama.from_pretrained.return_value = _fake_model()
        GGUFEmbeddings(config={"model": "labse", "remote_filename": "labse.Q4_K_M.gguf",
                               "n_ctx": 512, "n_gpu_layers": 2})

    _, kwargs = MockLlama.from_pretrained.call_args
    assert kwargs["n_ctx"] == 512
    assert kwargs["n_gpu_layers"] == 2          # overrides the 0 default
    assert kwargs["embedding"] is True
    assert kwargs["verbose"] is False           # default applied
    assert "model" not in kwargs                # config-only keys not forwarded
    assert "remote_filename" not in kwargs


def test_shared_engine_skips_loading():
    fake = _fake_model(vector=(1.0, 2.0))
    with mock.patch("ovos_gguf_plugin.embeddings.Llama") as MockLlama:
        emb = GGUFEmbeddings(config={"model": "labse"}, gguf_engine=fake)
        MockLlama.assert_not_called()
        MockLlama.from_pretrained.assert_not_called()
    assert np.allclose(emb.get_embeddings("x"), [1.0, 2.0])


def test_load_failure_leaves_model_none_and_raises():
    with mock.patch("ovos_gguf_plugin.embeddings.Llama") as MockLlama, \
            mock.patch("ovos_gguf_plugin.embeddings.os.path.isfile", return_value=False):
        MockLlama.from_pretrained.side_effect = RuntimeError("boom")
        emb = GGUFEmbeddings(config={"model": "labse"})

    assert emb.model is None
    try:
        emb.get_embeddings("hello")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_registry_is_populated():
    assert "labse" in GGUFEmbeddings.DEFAULT_MODELS
    assert len(GGUFEmbeddings.DEFAULT_MODELS) >= 20
    for name, loc in GGUFEmbeddings.DEFAULT_MODELS.items():
        assert isinstance(loc, tuple) and len(loc) == 2  # (repo_id, filename)
