from typing import Optional, Dict, Set

from langcodes import Language
from ovos_config import Configuration
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_plugin_manager.templates.language import LanguageTranslator, LanguageDetector
from ovos_utils import classproperty
from ovos_spec_tools import standardize_lang
from ovos_gguf_plugin.chat import GGUFChatEngine, Llama
from ovos_gguf_plugin.prompts import load_prompt, default_lang


class GGUFTextTranslator(LanguageTranslator):
    def __init__(self, config: Optional[Dict[str, str]] = None,
                 gguf_engine: Optional[Llama] = None):
        super().__init__(config)
        if "system_prompt" not in self.config:
            self.config["system_prompt"] = load_prompt("translate_system", default_lang())
        if "model" not in self.config:
            self.config["model"] = "TheBloke/TowerInstruct-7B-v0.1-GGUF"
        if "remote_filename" not in self.config:
            self.config["remote_filename"] = "*Q4_K_M.gguf"
        if "n_gpu_layers" not in self.config:
            self.config["n_gpu_layers"] = -1
        self.api = GGUFChatEngine(self.config, gguf_engine=gguf_engine)

    @property
    def system_prompt(self):
        return self.api.system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        self.api.system_prompt = value

    def translate(self, text: str, target: Optional[str] = None, source: Optional[str] = None) -> str:
        """
        Translate the given text from the source language to the target language.

        Args:
            text (str): The text to translate.
            target (Optional[str]): The target language code. If None, the internal language is used.
            source (Optional[str]): The source language code. If None, the default language is used.

        Returns:
            str: The translated text.
        """
        target = target or Configuration()["lang"]
        tgt = Language.get(target).display_name('en')
        lang = default_lang()
        if source:
            src = Language.get(source).display_name('en')
            prompt = load_prompt("translate_with_source", lang,
                                 {"source": src, "target": tgt, "text": text})
        else:
            prompt = load_prompt("translate_no_source", lang,
                                 {"target": tgt, "text": text})
        return self.api.continue_chat([
            AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            AgentMessage(role=MessageRole.USER, content=prompt)
        ]).content


class GGUFTextLangDetector(LanguageDetector):
    def __init__(self, config: Optional[Dict[str, str]] = None,
                 gguf_engine: Optional[Llama] = None):
        super().__init__(config)
        if "system_prompt" not in self.config:
            self.config["system_prompt"] = load_prompt("detect_system", default_lang())
        if "remote_filename" not in self.config:
            self.config["remote_filename"] = "*Q4_K_M.gguf"
        if "n_gpu_layers" not in self.config:
            self.config["n_gpu_layers"] = -1
        self.api = GGUFChatEngine(self.config, gguf_engine=gguf_engine)

    @property
    def system_prompt(self):
        return self.api.system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        self.api.system_prompt = value

    def detect(self, text):
        return standardize_lang(self.api.continue_chat([
            AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            AgentMessage(role=MessageRole.USER, content=load_prompt("detect_user", default_lang(), {"text": text}))
        ]).content)

    def detect_probs(self, text):
        l = self.detect(text)
        if l:
            return {l: 1.0}
        return {}

    @classproperty
    def available_languages(cls) -> Set[str]:
        """
        Return languages supported by this detector implementation in this state.
        This should be a set of languages this detector is capable of recognizing.
        This property should be overridden by the derived class to advertise
        what languages that engine supports.
        Returns:
            Set[str]: A set of language codes supported by this detector.
        """
        return set()  # TODO


if __name__ == "__main__":
    cfg = {
        "model": "Qwen/Qwen2-0.5B-Instruct-GGUF",
        "remote_filename": "*q8_0.gguf"
    }
    dt = GGUFTextLangDetector(config=cfg)

    print(dt.detect(
        "The easiest way for anyone to contribute is to help with translations! You can help without any programming knowledge via the translation portal"
    ))
    # en

    cfg = {
        "model": "TheBloke/TowerInstruct-7B-v0.1-GGUF",
        "remote_filename": "*Q4_K_M.gguf"
    }
    tx = GGUFTextTranslator(config=cfg)

    print(tx.translate(
        "The easiest way for anyone to contribute is to help with translations! You can help without any programming knowledge via the translation portal",
        target="es-es"))
    # La forma más sencilla de contribuir es ayudando con las traducciones! Puedes ayudar sin conocimientos de programación a través del portal de traducción


    print(tx.translate(
        "(how|what) is the weather [like] [tomorrow] in {location}",
        target="es-es"))
    # ¿Cómo (o qué) es el clima [como] [mañana] en {ubicación}

    tx.system_prompt = """You are a professional translator. 
    Your task is to translate OpenVoiceOS .intent files.

    RULES:
     - each sentence may contain variables between curly braces -> {variable}
     - the {variable_name} inside should NEVER be modified, only translate surrounding text
     - optional words may be indicated between square brackets -> [optional text]
     - alternative text can be indicated with parenthesis and | -> (this|that)
     - you can modify/add/remove optional/alternative words depending on if they make sense for the target language
     - you can NOT modify/add/remove variables, only change the position inside the sentence
     - if there are multiple variables in a sentence, there MUST be some text between them
    """
    print(tx.translate(
        "(how|what) is the weather [like] [tomorrow] in {location}",
        target="es-es"))
    # ¿Cómo es el tiempo [en] [mañana] en {location}

