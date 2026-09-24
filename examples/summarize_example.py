"""Summarization with a tiny GGUF model."""
from ovos_gguf_plugin.summarizer import GGUFSummarizer

s = GGUFSummarizer({"model": "Qwen/Qwen2-0.5B-Instruct-GGUF",
                    "remote_filename": "*q8_0.gguf"})
print(s.summarize("Long document text goes here ... " * 20))
