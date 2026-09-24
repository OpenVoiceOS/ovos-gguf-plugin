"""Text embeddings with a small GGUF model (downloads on first run)."""
from ovos_gguf_plugin.embeddings import GGUFEmbeddings

emb = GGUFEmbeddings({"model": "all-MiniLM-L6-v2"})
vec = emb.get_embeddings("the quick brown fox")
print(f"{len(vec)} dims:", vec[:4], "...")
