# Configuration

Every wrapper loads a quantized GGUF model through `llama-cpp-python` and shares a
common config contract. A model may be a Hugging Face repo (downloaded on first
use) or a local `.gguf` path.

## Common keys

| Key | Description | Default |
|---|---|---|
| `model` | HF repo id, or a local `.gguf` path. For embeddings, also a `DEFAULT_MODELS` friendly name. | per wrapper |
| `remote_filename` | file glob to fetch from the HF repo | `*Q4_K_M.gguf` |
| `n_gpu_layers` | layers to offload to GPU (`-1` = all) | `0` |
| `verbose` | llama.cpp verbose logging | varies |
| `system_prompt` | overrides the localized system prompt (see [localization](localization.md)) | from `.prompt` |

GPU support requires llama.cpp built with CUDA:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## Per-wrapper notes

- **chat** (`GGUFChatEngine`, `opm.agents.chat`): `max_tokens`, `chat_format`,
  `allow_system_prompts`, `drop_incomplete_sentences`, `end_markers`.
- **summarizer** (`GGUFSummarizer`, `opm.agents.summarizer`): `prompt_template`
  (an explicit `{content}` template) overrides the localized `summarize_user` prompt.
- **translate** (`GGUFTextTranslator`, `opm.lang.translate`): defaults to
  `TheBloke/TowerInstruct-7B-v0.1-GGUF`.
- **lang detect** (`GGUFTextLangDetector`, `opm.lang.detect`).
- **dialog transformer** (`GGUFDialogTransformer`, `opm.transformer.dialog`): the
  per-call rewrite instruction comes from `context["prompt"]` or `config["rewrite_prompt"]`.
- **embeddings** (`GGUFEmbeddings`, `opm.embeddings.text`): `model` may be a
  `GGUFEmbeddings.DEFAULT_MODELS` name (for example `labse`, `all-MiniLM-L6-v2`,
  `nomic-embed-text-v1.5`). The default is `labse`. It pairs with an `EmbeddingsDB`
  vector store (`ovos-chromadb-embeddings-plugin`, `ovos-qdrant-embeddings-plugin`).

A single loaded model can be shared across wrappers by passing `gguf_engine=`.

---
[Home](../README.md) · [Localized prompts →](localization.md)
