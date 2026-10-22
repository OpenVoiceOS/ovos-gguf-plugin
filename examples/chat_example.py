"""Conversational chat with a tiny GGUF model (TinyLlama)."""
from ovos_gguf_plugin.chat import GGUFChatEngine
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

bot = GGUFChatEngine({"model": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                      "remote_filename": "*Q4_K_M.gguf"})
msgs = [AgentMessage(role=MessageRole.USER, content="tell me a joke about aliens")]
for sentence in bot.stream_sentences(msgs):
    print(sentence)
