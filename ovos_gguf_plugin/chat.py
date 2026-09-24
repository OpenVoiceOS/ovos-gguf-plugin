import os
import re
from typing import Dict, Optional, List, Any, Iterable
from sentence_stream import SentenceBoundaryDetector

from llama_cpp import Llama
from ovos_plugin_manager.templates.agents import ChatEngine, AgentMessage, MessageRole, ToolsArg
from ovos_utils.log import LOG

#: Chat-template end markers a model can emit as literal TEXT instead of as
#: its end-of-sequence token. A small or heavily quantised build does this:
#: `smol-llama-101m-chat` at q2_k streams `<|im_end|>` as characters, and
#: nothing downstream removes it, so it is spoken aloud.
#:
#: It also hides a sentence boundary. `SentenceBoundaryDetector` needs a
#: terminator followed by a space, and a marker glued to the full stop gives
#: it neither, so `"...second part.<|im_end|>"` never reports a boundary and
#: leaves in `finish()` as one string with the marker attached.
END_MARKERS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>", "</s>")


def strip_end_markers(chunks: Iterable[str],
                      markers: Iterable[str] = END_MARKERS) -> Iterable[str]:
    """Yield the stream with chat-template end markers removed.

    A marker can arrive split across chunks, so text that could still be the
    start of one is held back until the next chunk settles it. Whatever is
    held at the end is yielded: a partial marker is only a partial marker
    once the stream is over, and dropping real text would be worse.
    """
    # Handle string as a single marker, and empty list as no stripping
    markers = (markers,) if isinstance(markers, str) else tuple(markers)
    if not markers:
        # Empty marker set: pass through unchanged
        yield from chunks
        return
    longest = max(len(m) for m in markers)
    buffer = ""
    for chunk in chunks:
        buffer += chunk
        for marker in markers:
            buffer = buffer.replace(marker, "")
        # keep back only as much as could still grow into a marker
        hold = 0
        for size in range(1, min(longest, len(buffer)) + 1):
            tail = buffer[-size:]
            if any(m.startswith(tail) for m in markers):
                hold = size
        if hold:
            out, buffer = buffer[:-hold], buffer[-hold:]
        else:
            out, buffer = buffer, ""
        if out:
            yield out
    if buffer:
        for marker in markers:
            buffer = buffer.replace(marker, "")
        if buffer:
            yield buffer


#: A sentence the model finished, as opposed to one max_tokens cut short.
#: ``SentenceBoundaryDetector`` reports a boundary only when another sentence
#: starts after it, so the last sentence of every answer stays in ``finish()``
#: and is complete far more often than not. Dropping it unread loses the end
#: of the answer, and for a one-sentence answer loses all of it.
SENTENCE_COMPLETE = re.compile(
    r"""[.!?\u2026\u3002\uff01\uff1f]['")\]}\u00bb\u201d\u2019]*\s*$""")


class GGUFChatEngine(ChatEngine):
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 gguf_engine: Optional[Llama] = None):
        config = config or {}
        super().__init__(config)
        if gguf_engine:
            self.model = gguf_engine
        else:
            if "model" not in self.config:
                raise ValueError("no 'model' set in config")
            model = self.config["model"]
            if os.path.isfile(model):  # local path
                LOG.info(f"Loading GGUF model: {model}")
                self.model = Llama(
                    model_path=model,
                    n_gpu_layers=self.config.get("n_gpu_layers", 0),
                    chat_format=self.config.get("chat_format"),
                    verbose=self.config.get("verbose", True))
            else:
                fname = self.config.get("remote_filename", "*Q4_K_M.gguf")
                LOG.info(f"Loading GGUF model from hub: {model} from file: {fname}")
                self.model = Llama.from_pretrained(
                    repo_id=model,
                    filename=fname,
                    n_gpu_layers=self.config.get("n_gpu_layers", 0),
                    chat_format=self.config.get("chat_format"),
                    verbose=self.config.get("verbose", True)
                )
            LOG.info("GGUF model loaded!")
        self.system_prompt = self.config.get("system_prompt")
        self.allow_system = self.config.get("allow_system_prompts") or False

    def validate_messages(self, messages: List[AgentMessage]) -> List[AgentMessage]:
        """
        Prepares the message list by enforcing system prompt rules.

        This method:
        1. Strips existing system messages if `allow_system` is False.
        2. Injects the configured `system_prompt` if it exists.
        3. Merges the configured system prompt with an existing one if
           `allow_system` is True.

        Args:
            messages (List[AgentMessage]): The raw input history of messages.

        Returns:
            List[AgentMessage]: The processed list of messages ready for the API.
        """
        if not self.allow_system:
            messages = [m for m in messages if m.role != MessageRole.SYSTEM]

        if not messages:
            if self.system_prompt:
                return [AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt)]
            return []

        if self.system_prompt:
            sysm = AgentMessage(role=MessageRole.SYSTEM, content=self.system_prompt)
            if messages and messages[0].role == MessageRole.SYSTEM:
                if self.allow_system:  # merge system prompts
                    sysm = AgentMessage(role=MessageRole.SYSTEM,
                                        content=self.system_prompt + "\n" + messages[0].content)
                # replace system prompt
                messages[0] = sysm
            else:
                messages.insert(0, sysm)
        return messages

    ###########################################################
    # abstract methods to be implemented by individual plugins
    ###########################################################
    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None,
                      tools: "ToolsArg" = None) -> AgentMessage:
        """
        Generate a response message based on the provided chat history.

        Args:
            messages (List[AgentMessage]): Full list of messages in the conversation.
            session_id (str): Identifier for the session.
            lang (str, optional): BCP-47 language code.
            units (str, optional): Preferred unit system (e.g., "metric", "imperial").
            tools: unused. Accepted for contract conformance with
                ovos_plugin_manager.templates.agents.ChatEngine.continue_chat,
                whose base signature declares it unconditionally. GGUFChatEngine
                is not tool-capable (supports_tools stays False); omitting this
                parameter would raise TypeError whenever a caller (e.g.
                ovos_persona_server/server_tools.py, or the ovos-agentic-loop
                ReAct fallback that calls any configured brain with tools=)
                passes tools= by keyword.

        Returns:
            AgentMessage: The generated response message from the assistant.
        """
        ans = self.model.create_chat_completion(
            messages=[
                {"role": m.role, "content": m.content}
                for m in self.validate_messages(messages)
            ],
            max_tokens=self.config.get("max_tokens"),
            stream=False
        )["choices"][0]["message"]
        return AgentMessage(role=MessageRole.ASSISTANT, content=ans["content"])

    def stream_tokens(self, messages: List[AgentMessage],
                    session_id: str = "default",
                    lang: Optional[str] = None,
                    units: Optional[str] = None) -> Iterable[str]:
        """
        Stream back response tokens as they are generated.

        Returns partial sentences and is not suitable for direct TTS.

        Once merged the output corresponds to the content of a AgentMessage with MessageRole.ASSISTANT

        Note:
            Default implementation yields the full response from continue_chat.
            Subclasses should override this for real-time token streaming.

        Args:
            messages (List[AgentMessage]): Full list of messages.
            session_id (str): Identifier for the session.
            lang (str, optional): Language code.
            units (str, optional): Unit system.

        Returns:
            Iterable[str]: A stream of tokens/partial text.
        """
        # With stream=True, the output is of type `Iterator[CompletionChunk]`.
        ans = self.model.create_chat_completion(
            messages=[
                {"role": m.role, "content": m.content}
                for m in self.validate_messages(messages)
            ],
            max_tokens=self.config.get("max_tokens"),
            stream=True
        )
        def _chunks():
            for item in ans:
                chunk = item['choices'][0]["delta"].get("content")
                if chunk:
                    yield chunk

        # The markers are stripped HERE, not in stream_sentences, so every
        # caller of stream_tokens is covered and the boundary detector sees
        # the terminator with nothing glued to it.
        end_markers = self.config.get("end_markers", END_MARKERS)
        yield from strip_end_markers(
            _chunks(), END_MARKERS if end_markers is None else end_markers)

    def stream_sentences(self, messages: List[AgentMessage],
                    session_id: str = "default",
                    lang: Optional[str] = None,
                    units: Optional[str] = None) -> Iterable[str]:
        """
        Stream back response sentences as they are generated.

        Returns full sentences only, suitable for direct TTS.

        Once merged the output corresponds to the content of a AgentMessage with MessageRole.ASSISTANT

        Note:
            Default implementation yields the full response from continue_chat.
            Subclasses should override this for real-time token streaming.

        Args:
            messages (List[AgentMessage]): Full list of messages.
            session_id (str): Identifier for the session.
            lang (str, optional): Language code.
            units (str, optional): Unit system.

        Returns:
            Iterable[str]: A stream of tokens/partial text.
        """
        boundary_detector = SentenceBoundaryDetector()
        spoke = False
        for tok in self.stream_tokens(messages):
            for sentence in boundary_detector.add_chunk(tok):
                spoke = True
                yield sentence
        final_text = boundary_detector.finish()
        if final_text and (SENTENCE_COMPLETE.search(final_text) or not spoke or
                           not self.config.get("drop_incomplete_sentences", True)):
            yield final_text


if __name__ == "__main__":
    LOG.set_level("DEBUG")

    cfg = {
        "model": "Qwen/Qwen2-0.5B-Instruct-GGUF",
        "remote_filename": "*q8_0.gguf"
    }

    query = """The possibility of alien life in the solar system has been a topic of interest for scientists and astronomers for many years. The search for extraterrestrial life has been a major focus of space exploration, with numerous missions and discoveries made in recent years. While there is still no concrete evidence of life beyond Earth, the search for alien life continues to be a fascinating and exciting endeavor.
One of the most promising areas for the search for alien life is the moons of Jupiter and Saturn. These moons, such as Europa and Enceladus, are believed to have subsurface oceans that could potentially harbor life. The presence of water, a key ingredient for life as we know it, has been detected on these moons, and there are also indications of other necessary elements such as carbon, nitrogen, and oxygen.
Another area of interest for the search for alien life is the asteroid belt between Mars and Jupiter. This region is home to millions of asteroids, some of which may have the right conditions for life to exist. For example, some asteroids have been found to have water and organic compounds, which are essential for life.
In addition to the moons and asteroids of the solar system, there are also other potential locations for the search for alien life. For example, there are exoplanets, or planets outside of our solar system, that have been discovered in recent years. Some of these exoplanets are believed to be in the habitable zone, which means they are located in the right distance from their star to potentially have liquid water on their surface.
Despite the potential for alien life in the solar system, there are still many uncertainties and unknowns. The search for extraterrestrial life is a complex and multifaceted endeavor that requires a combination of scientific research, technological advancements, and exploration. While there is still no concrete evidence of life beyond Earth, the search for alien life continues to be a fascinating and exciting endeavor that holds the potential for groundbreaking discoveries in the future."""
    s = GGUFChatEngine(cfg)
    messages = [AgentMessage(role=MessageRole.USER, content=query)]
    for sent in s.stream_sentences(messages):
        print(sent)