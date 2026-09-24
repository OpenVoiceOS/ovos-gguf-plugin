import os
from typing import Any, Dict, Optional

import numpy as np
from ovos_plugin_manager.templates.embeddings import EmbeddingsArray, TextEmbedder
from ovos_utils.log import LOG

from llama_cpp import Llama


def _is_hub_rate_limit(exc: BaseException) -> bool:
    """True for a transient HuggingFace hub rate-limit (HTTP 429) error.

    Detected without importing ``huggingface_hub`` directly: the hub raises
    ``HfHubHTTPError`` which carries a ``response`` with ``status_code``; we
    fall back to message sniffing for wrappers that drop the response.
    """
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc)
    return "429" in text and "Too Many Requests" in text


class GGUFEmbeddings(TextEmbedder):
    """Text embeddings via GGUF models executed through llama.cpp.

    Models load through huggingface-hub (repo id + filename) like the other
    wrappers in this plugin, or from a local ``.gguf`` path. ``DEFAULT_MODELS``
    maps friendly names to their hub location; any other ``model`` value is
    treated as a hub repo id (with ``remote_filename`` as the file glob) or a
    local file path. A loaded ``llama_cpp.Llama`` may be shared via
    ``gguf_engine`` so several wrappers reuse one model.
    """

    # friendly name -> (hugging-face repo id, filename)
    DEFAULT_MODELS = {
        "all-MiniLM-L6-v2": ("leliuga/all-MiniLM-L6-v2-GGUF", "all-MiniLM-L6-v2.Q4_K_M.gguf"),
        "all-MiniLM-L12-v2": ("leliuga/all-MiniLM-L12-v2-GGUF", "all-MiniLM-L12-v2.Q4_K_M.gguf"),
        "multi-qa-MiniLM-L6-cos-v1": ("Felladrin/gguf-multi-qa-MiniLM-L6-cos-v1", "multi-qa-MiniLM-L6-cos-v1.Q4_K_M.gguf"),
        "gist-all-minilm-l6-v2": ("afrideva/GIST-all-MiniLM-L6-v2-GGUF", "gist-all-minilm-l6-v2.Q4_K_M.gguf"),
        "e5-small-v2": ("ChristianAzinn/e5-small-v2-gguf", "e5-small-v2.Q4_K_M.gguf"),
        "gte-small": ("ChristianAzinn/gte-small-gguf", "gte-small.Q4_K_M.gguf"),
        "gte-base": ("ChristianAzinn/gte-base-gguf", "gte-base.Q4_K_M.gguf"),
        "gte-large": ("ChristianAzinn/gte-large-gguf", "gte-large.Q4_K_M.gguf"),
        "snowflake-arctic-embed-l": ("ChristianAzinn/snowflake-arctic-embed-l-gguf", "snowflake-arctic-embed-l--Q4_K_M.GGUF"),
        "snowflake-arctic-embed-m": ("ChristianAzinn/snowflake-arctic-embed-m-gguf", "snowflake-arctic-embed-m--Q4_K_M.GGUF"),
        "snowflake-arctic-embed-s": ("ChristianAzinn/snowflake-arctic-embed-s-gguf", "snowflake-arctic-embed-s--Q4_K_M.GGUF"),
        "snowflake-arctic-embed-xs": ("ChristianAzinn/snowflake-arctic-embed-xs-gguf", "snowflake-arctic-embed-xs--Q4_K_M.GGUF"),
        "nomic-embed-text-v1.5": ("nomic-ai/nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.Q4_K_M.gguf"),
        "uae-large-v1": ("ChristianAzinn/uae-large-v1-gguf", "uae-large-v1.Q4_K_M.gguf"),
        "labse": ("ChristianAzinn/labse-gguf", "labse.Q4_K_M.gguf"),
        "bge-large-en-v1.5": ("ChristianAzinn/bge-large-en-v1.5-gguf", "bge-large-en-v1.5.Q4_K_M.gguf"),
        "bge-base-en-v1.5": ("ChristianAzinn/bge-base-en-v1.5-gguf", "bge-base-en-v1.5.Q4_K_M.gguf"),
        "bge-small-en-v1.5": ("ChristianAzinn/bge-small-en-v1.5-gguf", "bge-small-en-v1.5.Q4_K_M.gguf"),
        "gist-large-embedding-v0": ("ChristianAzinn/gist-large-embedding-v0-gguf", "gist-large-embedding-v0.Q4_K_M.gguf"),
        "gist-embedding-v0": ("ChristianAzinn/gist-embedding-v0-gguf", "gist-embedding-v0.Q4_K_M.gguf"),
        "gist-small-embedding-v0": ("ChristianAzinn/gist-small-embedding-v0-gguf", "gist-small-embedding-v0.Q4_K_M.gguf"),
        "mxbai-embed-large-v1": ("ChristianAzinn/mxbai-embed-large-v1-gguf", "mxbai-embed-large-v1.Q4_K_M.gguf"),
        "acge_text_embedding": ("ChristianAzinn/acge_text_embedding-gguf", "acge_text_embedding-Q4_K_M.GGUF"),
        "gte-Qwen2-1.5B-instruct": ("second-state/gte-Qwen2-1.5B-instruct-GGUF", "gte-Qwen2-1.5B-instruct-Q4_K_M.gguf"),
        "gte-Qwen2-7B-instruct": ("niancheng/gte-Qwen2-7B-instruct-Q4_K_M-GGUF", "gte-qwen2-7b-instruct-q4_k_m.gguf"),
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 gguf_engine: Optional[Llama] = None):
        """Load (or reuse) a GGUF embeddings model.

        Args:
            config: Plugin config. ``model`` (default ``labse``) is a
                ``DEFAULT_MODELS`` name, a hub repo id, or a local ``.gguf``
                path; ``remote_filename`` is the hub file glob for bare repo
                ids; any other key is forwarded to ``llama_cpp.Llama``.
            gguf_engine: An already-loaded ``Llama`` (with ``embedding=True``)
                to reuse instead of loading a new one.
        """
        super().__init__(config)
        self.model = gguf_engine
        if self.model is None:
            self._load_model()

    def _load_model(self) -> None:
        model_id = self.config.get("model", "labse")
        llama_args = {k: v for k, v in self.config.items()
                      if k not in ("model", "remote_filename")}
        llama_args.setdefault("n_gpu_layers", 0)
        llama_args.setdefault("verbose", False)
        llama_args["embedding"] = True
        try:
            if os.path.isfile(model_id):
                LOG.info(f"Loading GGUF embeddings model: {model_id}")
                self.model = Llama(model_path=model_id, **llama_args)
            else:
                if model_id in self.DEFAULT_MODELS:
                    repo_id, filename = self.DEFAULT_MODELS[model_id]
                else:
                    repo_id = model_id
                    filename = self.config.get("remote_filename", "*Q4_K_M.gguf")
                LOG.info(f"Loading GGUF embeddings model from hub: {repo_id} ({filename})")
                self.model = Llama.from_pretrained(repo_id=repo_id, filename=filename, **llama_args)
            LOG.info("GGUF embeddings model loaded!")
        except Exception as e:
            # Transient HuggingFace rate-limiting (HTTP 429) is not a real
            # failure -- surface it so callers can retry with backoff instead
            # of seeing a permanently unloaded model.
            if _is_hub_rate_limit(e):
                LOG.warning(
                    f"HuggingFace rate-limited the download of '{model_id}' (429); "
                    f"re-raising so the caller can retry"
                )
                self.model = None
                raise
            LOG.error(f"Failed to load GGUF embeddings model '{model_id}': {e}")
            self.model = None

    def get_embeddings(self, text: str) -> EmbeddingsArray:
        """Return the embedding vector for ``text`` as a numpy array.

        Raises:
            RuntimeError: If the model failed to load.
        """
        if self.model is None:
            raise RuntimeError("Embedding model not loaded. Check logs for errors during initialization.")
        result = self.model.create_embedding(text)
        return np.array(result["data"][0]["embedding"])
