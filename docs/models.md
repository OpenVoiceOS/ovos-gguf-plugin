# Recommended Models

Models are loaded from the Hugging Face Hub by default. The `model` key accepts a hub repo id. `remote_filename` is a glob that selects the quantization variant. Local `.gguf` paths work too.

## Chat / Summarizer / Dialog / Lang-detect / Translate

These wrappers use generative chat models. Any instruction-tuned GGUF model works. Smaller models run faster on CPU.

### Tiny (CI / low-RAM, < 200 MB)

| Model | `model` | `remote_filename` | Notes |
|---|---|---|---|
| Smol-Llama 101M | `afrideva/Smol-Llama-101M-Chat-v1-GGUF` | `*q2_k.gguf` | ~45 MB, used in CI e2e tests |
| Lite-Mistral 150M | `OuteAI/Lite-Mistral-150M-v2-Instruct-GGUF` | `*Q4_K_M.gguf` | ~100 MB |

### Small (1 to 2 GB, good for Raspberry Pi 5)

| Model | `model` | `remote_filename` | Notes |
|---|---|---|---|
| Qwen2 0.5B Instruct | `Qwen/Qwen2-0.5B-Instruct-GGUF` | `*q8_0.gguf` | multilingual |
| TinyLlama 1.1B | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` | `*Q4_K_M.gguf` | English |

### Medium (4 to 8 GB)

| Model | `model` | `remote_filename` | Notes |
|---|---|---|---|
| Mistral 7B Instruct v0.3 | `MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF` | `*Q4_K_M.gguf` | strong general |
| Phi-3 Mini 4K | `microsoft/Phi-3-mini-4k-instruct-gguf` | `*Q4_K_M.gguf` | 3.8B, very capable |
| Qwen2 7B Instruct | `Qwen/Qwen2-7B-Instruct-GGUF` | `*Q4_K_M.gguf` | multilingual |

### Translation-specific

| Model | `model` | `remote_filename` | Notes |
|---|---|---|---|
| TowerInstruct 7B | `TheBloke/TowerInstruct-7B-v0.1-GGUF` | `*Q4_K_M.gguf` | fine-tuned for translation |

### Portuguese

| Model | `model` | `remote_filename` |
|---|---|---|
| Gervásio 7B PTPT | `RichardErkhov/PORTULAN_-_gervasio-7b-portuguese-ptpt-decoder-gguf` | `*Q4_K_M.gguf` |
| CabraLlama3 8B | `mradermacher/CabraLlama3-8b-GGUF` | `*Q4_K_M.gguf` |
| Bode 7B PT-BR | `recogna-nlp/bode-7b-alpaca-pt-br-gguf` | `*Q4_K_M.gguf` |

### Catalan

| Model | `model` | `remote_filename` |
|---|---|---|
| CataLlama v0.2 | `catallama/CataLlama-v0.2-Instruct-SFT-DPO-Merged-GGUF` | `*Q4_K_M.gguf` |

---

## Embeddings

`GGUFEmbeddings` supports friendly names from `GGUFEmbeddings.DEFAULT_MODELS`:

### Tiny (< 30 MB)

| Friendly name | Repo | Dims | Notes |
|---|---|---|---|
| `all-MiniLM-L6-v2` | `leliuga/all-MiniLM-L6-v2-GGUF` | 384 | used in CI e2e tests |
| `e5-small-v2` | `ChristianAzinn/e5-small-v2-gguf` | 384 | |
| `gte-small` | `ChristianAzinn/gte-small-gguf` | 384 | |

### Small (30 to 100 MB)

| Friendly name | Repo | Dims | Notes |
|---|---|---|---|
| `all-MiniLM-L12-v2` | `leliuga/all-MiniLM-L12-v2-GGUF` | 384 | |
| `bge-small-en-v1.5` | `ChristianAzinn/bge-small-en-v1.5-gguf` | 384 | English |
| `gte-base` | `ChristianAzinn/gte-base-gguf` | 768 | |
| `snowflake-arctic-embed-xs` | `ChristianAzinn/snowflake-arctic-embed-xs-gguf` | 384 | |
| `snowflake-arctic-embed-s` | `ChristianAzinn/snowflake-arctic-embed-s-gguf` | 384 | |

### Multilingual

| Friendly name | Repo | Dims | Notes |
|---|---|---|---|
| `labse` | `ChristianAzinn/labse-gguf` | 768 | 109 languages, default |
| `nomic-embed-text-v1.5` | `nomic-ai/nomic-embed-text-v1.5-GGUF` | 768 | |

### Large / high-quality

| Friendly name | Repo | Dims | Notes |
|---|---|---|---|
| `bge-large-en-v1.5` | `ChristianAzinn/bge-large-en-v1.5-gguf` | 1024 | English |
| `mxbai-embed-large-v1` | `ChristianAzinn/mxbai-embed-large-v1-gguf` | 1024 | |
| `uae-large-v1` | `ChristianAzinn/uae-large-v1-gguf` | 1024 | |
| `gte-large` | `ChristianAzinn/gte-large-gguf` | 1024 | |
| `gte-Qwen2-1.5B-instruct` | `second-state/gte-Qwen2-1.5B-instruct-GGUF` | 1536 | multilingual |

> These are community-maintained GGUF quantizations. Check the linked repos for licensing details before use in production.

---
[← Localized prompts](localization.md) · [Home](../README.md)
