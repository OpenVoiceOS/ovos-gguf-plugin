from typing import Tuple, Optional

from ovos_plugin_manager.templates.transformers import DialogTransformer
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_gguf_plugin.chat import GGUFChatEngine, Llama
from ovos_gguf_plugin.prompts import load_prompt, default_lang


class GGUFDialogTransformer(DialogTransformer):
    def __init__(self, name="ovos-dialog-transformer-gguf-plugin", priority=10, config=None,
                 gguf_engine: Optional[Llama] = None):
        """
        Initializes the OpenAIDialogTransformer with a name, priority, and configuration.
        
        Creates an OpenAIChatCompletionsSolver using the provided API key, API URL, and a system prompt from the configuration or a default prompt if not specified.
        """
        super().__init__(name, priority, config)
        if "system_prompt" not in self.config:
            self.config["system_prompt"] = load_prompt("dialog_transform_system", default_lang())
        self.api = GGUFChatEngine(config=self.config, gguf_engine=gguf_engine)

    @property
    def system_prompt(self):
        return self.api.system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        self.api.system_prompt = value

    def transform(self, dialog: str, context: dict = None) -> Tuple[str, dict]:
        """
        Transforms the dialog string using a character-specific prompt if available.
        
        If a prompt is provided in the context or configuration, rewrites the dialog as if spoken by a different character using the solver; otherwise, returns the original dialog unchanged.
        
        Args:
            dialog: The dialog string to be transformed.
            context: Optional dictionary containing transformation context, such as a prompt or language.
        
        Returns:
            A tuple containing the transformed (or original) dialog and the unchanged context.
        """
        context = context or {}
        prompt = context.get("prompt") or self.config.get("rewrite_prompt")
        if not prompt:
            return dialog, context
        return self.api.continue_chat([
            AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            AgentMessage(role=MessageRole.USER, content=f"{prompt} : {dialog}")
        ]).content, context
