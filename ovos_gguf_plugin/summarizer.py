from typing import Dict, Optional

from ovos_plugin_manager.templates.agents import SummarizerEngine
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_gguf_plugin.chat import GGUFChatEngine, Llama
from ovos_gguf_plugin.prompts import load_prompt, default_lang


class GGUFSummarizer(SummarizerEngine):
    def __init__(self, config: Optional[Dict] = None,
                 gguf_engine: Optional[Llama] = None):
        super().__init__(config=config)
        if "system_prompt" not in self.config:
            self.config["system_prompt"] = load_prompt("summarize_system", default_lang())
        self.api = GGUFChatEngine(config=self.config, gguf_engine=gguf_engine)
        # explicit override wins; otherwise the localized summarize_user .prompt is used
        self.prompt_template = self.config.get("prompt_template")

    @property
    def system_prompt(self):
        return self.api.system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        self.api.system_prompt = value

    def summarize(self, document: str, lang: Optional[str] = None) -> str:
        """
        Create a summary of the provided text.

        Args:
            document (str): The full text to be summarized.
            lang (str, optional): The language of the document.

        Returns:
            str: The summarized text.
        """
        if self.prompt_template:
            prompt = self.prompt_template.format(content=document)
        else:
            prompt = load_prompt("summarize_user", lang or default_lang(), {"content": document})
        return self.api.continue_chat([
            AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            AgentMessage(role=MessageRole.USER, content=prompt)
        ]).content
