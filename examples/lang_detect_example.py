"""Language detection with a tiny GGUF model."""
from ovos_gguf_plugin.translate import GGUFTextLangDetector

dt = GGUFTextLangDetector({"model": "Qwen/Qwen2-0.5B-Instruct-GGUF",
                           "remote_filename": "*q8_0.gguf"})
print(dt.detect("you can help without any programming knowledge"))  # en
