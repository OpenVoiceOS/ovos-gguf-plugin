"""Machine translation with a GGUF translation model."""
from ovos_gguf_plugin.translate import GGUFTextTranslator

tx = GGUFTextTranslator({"model": "TheBloke/TowerInstruct-7B-v0.1-GGUF",
                         "remote_filename": "*Q4_K_M.gguf"})
print(tx.translate("the easiest way to contribute is to help with translations",
                   target="es-es"))
